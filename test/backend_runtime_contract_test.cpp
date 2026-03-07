#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/platform/backend_build.hpp"
#include "tinygs/platform/runtime_factory.hpp"

namespace {

using namespace tinygs;

bool has_cuda_device() {
  try {
    return cuda_device_count() > 0;
  } catch (...) {
    return false;
  }
}

BackendConfig make_cuda_config() {
  BackendConfig config;
  config.type = compiled_backend_type();
  config.device = 0;
  return config;
}

// Decorator that counts sync calls and delegates to an inner runtime.
class CountingRuntime final : public BackendRuntime {
public:
  explicit CountingRuntime(std::shared_ptr<BackendRuntime> inner) : m_inner(std::move(inner)) {}

  int sync_queue_calls = 0;
  int sync_device_calls = 0;

  BackendType backend_type() const noexcept override { return m_inner->backend_type(); }
  int device() const noexcept override { return m_inner->device(); }
  CapabilityProfile capability_profile() const override { return m_inner->capability_profile(); }

protected:
  Result<BackendQueue> do_create_queue(const QueueDesc& desc) override {
    return m_inner->create_queue(desc);
  }
  Result<BackendEvent> do_create_event(const EventDesc& desc) override {
    return m_inner->create_event(desc);
  }
  Result<BackendBuffer> do_create_buffer(const BufferDesc& desc) override {
    return m_inner->create_buffer(desc);
  }

  BackendError do_record_event(BackendQueue& queue, BackendEvent& event) override {
    return m_inner->record_event(queue, event);
  }
  BackendError do_wait_event(BackendQueue& queue, BackendEvent& event) override {
    return m_inner->wait_event(queue, event);
  }

  BackendError do_synchronize_queue(BackendQueue& queue) override {
    sync_queue_calls += 1;
    return m_inner->synchronize_queue(queue);
  }
  BackendError do_synchronize_event(BackendEvent& event) override {
    return m_inner->synchronize_event(event);
  }
  BackendError do_synchronize_device() override {
    sync_device_calls += 1;
    return m_inner->synchronize_device();
  }

  BackendError do_copy_buffer(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    return m_inner->copy_buffer_async(queue, dst, src, CopyRegion{size_bytes, src_offset, dst_offset});
  }
  BackendError do_copy_from_host(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      const void* src, size_t size_bytes) override {
    return m_inner->copy_from_host_async(queue, dst, src, BufferTransferRegion{size_bytes, dst_offset});
  }
  BackendError do_copy_to_host(
      BackendQueue& queue,
      void* dst,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    return m_inner->copy_to_host_async(queue, dst, src, BufferTransferRegion{size_bytes, src_offset});
  }
  BackendError do_transfer_raw(
      BackendQueue& queue,
      void* dst, const void* src,
      size_t size_bytes,
      TransferDirection direction) override {
    if (direction == TransferDirection::DeviceToHost) {
      return m_inner->copy_device_to_host_async(queue, dst, src, size_bytes);
    }
    return m_inner->copy_host_to_device_async(queue, dst, src, size_bytes);
  }
  BackendError do_fill_buffer(
      BackendQueue& queue,
      BackendBuffer& buffer,
      size_t offset, uint8_t value, size_t size_bytes) override {
    return m_inner->fill_buffer_async(queue, buffer, value, offset, size_bytes);
  }

private:
  std::shared_ptr<BackendRuntime> m_inner;
};

TEST(BackendRuntimeFactoryTest, MismatchedBackendReturnsInvalidArgument) {
  BackendConfig config = make_cuda_config();
  if (compiled_backend_type() == BackendType::Cuda) {
    config.type = BackendType::Hip;
  } else {
    config.type = BackendType::Cuda;
  }
  const auto runtime_result = create_backend_runtime(config);
  EXPECT_FALSE(runtime_result.ok());
  EXPECT_EQ(runtime_result.error().code, BackendErrorCode::InvalidArgument);
}

TEST(BackendRuntimeFactoryTest, CreatesRuntimeForCompiledBackend) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const auto runtime_result = create_backend_runtime(make_cuda_config());
  ASSERT_TRUE(runtime_result.ok()) << to_string(runtime_result.error());
  auto runtime = runtime_result.value();
  ASSERT_NE(runtime, nullptr);
  EXPECT_EQ(runtime->backend_type(), compiled_backend_type());
  EXPECT_EQ(runtime->device(), 0);
}

TEST(BackendRuntimeContractTest, CapabilityProfileIsPopulated) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const auto runtime_result = create_backend_runtime(make_cuda_config());
  ASSERT_TRUE(runtime_result.ok()) << to_string(runtime_result.error());
  auto runtime = runtime_result.value();
  const CapabilityProfile profile = runtime->capability_profile();
  EXPECT_TRUE(profile.supports_queues);
  EXPECT_TRUE(profile.supports_events);
  EXPECT_TRUE(profile.supports_device_buffers);
  EXPECT_GT(profile.compute_capability, 0U);
  EXPECT_GT(profile.total_global_memory_bytes, 0U);
}

TEST(BackendRuntimeContractTest, QueueEventAndBufferCopyFlow) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const auto runtime_result = create_backend_runtime(make_cuda_config());
  ASSERT_TRUE(runtime_result.ok()) << to_string(runtime_result.error());
  auto runtime = runtime_result.value();
  ASSERT_NE(runtime, nullptr);

  std::shared_ptr<BackendQueue> produce_queue;
  std::shared_ptr<BackendQueue> consume_queue;
  std::shared_ptr<BackendEvent> ready_event;
  std::shared_ptr<BackendBuffer> src_buffer;
  std::shared_ptr<BackendBuffer> dst_buffer;

  QueueDesc queue_desc;
  queue_desc.non_blocking = true;

  auto queue_result = runtime->create_queue(queue_desc);
  ASSERT_TRUE(queue_result.ok()) << to_string(queue_result.error());
  produce_queue = queue_result.value();
  queue_result = runtime->create_queue(queue_desc);
  ASSERT_TRUE(queue_result.ok()) << to_string(queue_result.error());
  consume_queue = queue_result.value();

  EventDesc event_desc;
  auto event_result = runtime->create_event(event_desc);
  ASSERT_TRUE(event_result.ok()) << to_string(event_result.error());
  ready_event = event_result.value();

  BufferDesc buffer_desc;
  buffer_desc.size_bytes = sizeof(int) * 32;
  buffer_desc.memory_class = BufferMemoryClass::Device;
  auto buffer_result = runtime->create_buffer(buffer_desc);
  ASSERT_TRUE(buffer_result.ok()) << to_string(buffer_result.error());
  src_buffer = buffer_result.value();
  buffer_result = runtime->create_buffer(buffer_desc);
  ASSERT_TRUE(buffer_result.ok()) << to_string(buffer_result.error());
  dst_buffer = buffer_result.value();

  std::vector<int> in_values(32);
  std::iota(in_values.begin(), in_values.end(), 7);
  BufferTransferRegion host_to_device_region;
  host_to_device_region.size_bytes = buffer_desc.size_bytes;
  BackendError status = runtime->copy_from_host_async(
      *produce_queue, *src_buffer, in_values.data(), host_to_device_region);
  ASSERT_TRUE(status.ok()) << to_string(status);

  status = runtime->record_event(*produce_queue, *ready_event);
  ASSERT_TRUE(status.ok()) << to_string(status);
  status = runtime->wait_event(*consume_queue, *ready_event);
  ASSERT_TRUE(status.ok()) << to_string(status);

  CopyRegion device_to_device_region;
  device_to_device_region.size_bytes = buffer_desc.size_bytes;
  status = runtime->copy_buffer_async(
      *consume_queue, *dst_buffer, *src_buffer, device_to_device_region);
  ASSERT_TRUE(status.ok()) << to_string(status);

  std::vector<int> out_values(32, -1);
  BufferTransferRegion device_to_host_region;
  device_to_host_region.size_bytes = buffer_desc.size_bytes;
  status = runtime->copy_to_host_async(
      *consume_queue, out_values.data(), *dst_buffer, device_to_host_region);
  ASSERT_TRUE(status.ok()) << to_string(status);
  status = runtime->synchronize_queue(*consume_queue);
  ASSERT_TRUE(status.ok()) << to_string(status);

  EXPECT_EQ(out_values, in_values);
}

TEST(BackendRuntimeContractTest, DeviceBufferHostAccessIsUnsupported) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const auto runtime_result = create_backend_runtime(make_cuda_config());
  ASSERT_TRUE(runtime_result.ok()) << to_string(runtime_result.error());
  auto runtime = runtime_result.value();
  ASSERT_NE(runtime, nullptr);

  BufferDesc buffer_desc;
  buffer_desc.size_bytes = sizeof(float) * 8;
  buffer_desc.memory_class = BufferMemoryClass::Device;
  buffer_desc.host_access = BufferHostAccess::ReadWrite;
  const auto buffer_result = runtime->create_buffer(buffer_desc);
  EXPECT_FALSE(buffer_result.ok());
  EXPECT_EQ(buffer_result.error().code, BackendErrorCode::Unsupported);
}

TEST(BackendRuntimeContractTest, CopyRegionOverflowIsInvalidArgument) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const auto runtime_result = create_backend_runtime(make_cuda_config());
  ASSERT_TRUE(runtime_result.ok()) << to_string(runtime_result.error());
  auto runtime = runtime_result.value();
  ASSERT_NE(runtime, nullptr);

  QueueDesc queue_desc;
  auto queue_result = runtime->create_queue(queue_desc);
  ASSERT_TRUE(queue_result.ok()) << to_string(queue_result.error());
  auto queue = queue_result.value();
  ASSERT_NE(queue, nullptr);

  BufferDesc buffer_desc;
  buffer_desc.size_bytes = sizeof(float) * 8;
  buffer_desc.memory_class = BufferMemoryClass::Device;
  auto src_result = runtime->create_buffer(buffer_desc);
  ASSERT_TRUE(src_result.ok()) << to_string(src_result.error());
  auto dst_result = runtime->create_buffer(buffer_desc);
  ASSERT_TRUE(dst_result.ok()) << to_string(dst_result.error());

  CopyRegion region;
  region.size_bytes = 16;
  region.dst_offset = std::numeric_limits<size_t>::max() - 8;
  region.src_offset = 0;
  const BackendError status = runtime->copy_buffer_async(
      *queue, *dst_result.value(), *src_result.value(), region);
  EXPECT_EQ(status.code, BackendErrorCode::InvalidArgument);
}

TEST(BackendRuntimeContractTest, GPUGaussianAsyncCopiesDoNotSynchronizeImplicitly) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  const auto inner_runtime_result = create_backend_runtime(make_cuda_config());
  ASSERT_TRUE(inner_runtime_result.ok()) << to_string(inner_runtime_result.error());
  auto runtime = std::make_shared<CountingRuntime>(inner_runtime_result.value());

  QueueDesc queue_desc;
  queue_desc.non_blocking = true;
  queue_desc.debug_name = "gpu_gaussian_async_test";
  const auto queue_result = runtime->create_queue(queue_desc);
  ASSERT_TRUE(queue_result.ok()) << to_string(queue_result.error());
  auto queue = queue_result.value();

  Gaussian3d in;
  in.means = {vec3(1.0f, 2.0f, 3.0f), vec3(-1.0f, 0.5f, 4.0f)};
  in.opacities = {0.1f, -0.3f};
  in.rotations = {vec4(1.0f, 0.0f, 0.0f, 0.0f), vec4(0.7f, 0.2f, 0.1f, 0.0f)};
  in.scales = {vec3(0.0f, 0.1f, 0.2f), vec3(-0.2f, 0.3f, 0.4f)};
  in.sh0 = {vec3(0.5f, 0.6f, 0.7f), vec3(0.8f, 0.9f, 1.0f)};
  in.sh1.resize(2 * 3, vec3(0.1f, 0.2f, 0.3f));
  in.sh2.resize(2 * 5, vec3(0.4f, 0.5f, 0.6f));
  in.sh3.resize(2 * 7, vec3(0.7f, 0.8f, 0.9f));

  GPUGaussian3d gpu(*runtime);
  gpu.copy_from_host_async(in, queue.get());
  EXPECT_EQ(runtime->sync_queue_calls, 0);

  Gaussian3d out;
  gpu.copy_to_host_async(out, queue.get());
  EXPECT_EQ(runtime->sync_queue_calls, 0);

  const BackendError sync_status = runtime->synchronize_queue(*queue);
  ASSERT_TRUE(sync_status.ok()) << to_string(sync_status);
  EXPECT_EQ(runtime->sync_queue_calls, 1);
  EXPECT_EQ(runtime->sync_device_calls, 0);

  ASSERT_EQ(out.means.size(), in.means.size());
  for (size_t i = 0; i < in.means.size(); ++i) {
    EXPECT_FLOAT_EQ(out.means[i].x, in.means[i].x);
    EXPECT_FLOAT_EQ(out.means[i].y, in.means[i].y);
    EXPECT_FLOAT_EQ(out.means[i].z, in.means[i].z);
    EXPECT_FLOAT_EQ(out.opacities[i], in.opacities[i]);
    EXPECT_FLOAT_EQ(out.rotations[i].x, in.rotations[i].x);
    EXPECT_FLOAT_EQ(out.rotations[i].y, in.rotations[i].y);
    EXPECT_FLOAT_EQ(out.rotations[i].z, in.rotations[i].z);
    EXPECT_FLOAT_EQ(out.rotations[i].w, in.rotations[i].w);
    EXPECT_FLOAT_EQ(out.scales[i].x, in.scales[i].x);
    EXPECT_FLOAT_EQ(out.scales[i].y, in.scales[i].y);
    EXPECT_FLOAT_EQ(out.scales[i].z, in.scales[i].z);
    EXPECT_FLOAT_EQ(out.sh0[i].x, in.sh0[i].x);
    EXPECT_FLOAT_EQ(out.sh0[i].y, in.sh0[i].y);
    EXPECT_FLOAT_EQ(out.sh0[i].z, in.sh0[i].z);
  }
  ASSERT_EQ(out.sh1.size(), in.sh1.size());
  ASSERT_EQ(out.sh2.size(), in.sh2.size());
  ASSERT_EQ(out.sh3.size(), in.sh3.size());
  for (size_t i = 0; i < in.sh1.size(); ++i) {
    EXPECT_FLOAT_EQ(out.sh1[i].x, in.sh1[i].x);
    EXPECT_FLOAT_EQ(out.sh1[i].y, in.sh1[i].y);
    EXPECT_FLOAT_EQ(out.sh1[i].z, in.sh1[i].z);
  }
  for (size_t i = 0; i < in.sh2.size(); ++i) {
    EXPECT_FLOAT_EQ(out.sh2[i].x, in.sh2[i].x);
    EXPECT_FLOAT_EQ(out.sh2[i].y, in.sh2[i].y);
    EXPECT_FLOAT_EQ(out.sh2[i].z, in.sh2[i].z);
  }
  for (size_t i = 0; i < in.sh3.size(); ++i) {
    EXPECT_FLOAT_EQ(out.sh3[i].x, in.sh3[i].x);
    EXPECT_FLOAT_EQ(out.sh3[i].y, in.sh3[i].y);
    EXPECT_FLOAT_EQ(out.sh3[i].z, in.sh3[i].z);
  }
}

}  // namespace
