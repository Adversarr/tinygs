#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

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
      produce_queue, src_buffer, in_values.data(), host_to_device_region);
  ASSERT_TRUE(status.ok()) << to_string(status);

  status = runtime->record_event(produce_queue, ready_event);
  ASSERT_TRUE(status.ok()) << to_string(status);
  status = runtime->wait_event(consume_queue, ready_event);
  ASSERT_TRUE(status.ok()) << to_string(status);

  CopyRegion device_to_device_region;
  device_to_device_region.size_bytes = buffer_desc.size_bytes;
  status = runtime->copy_buffer_async(
      consume_queue, dst_buffer, src_buffer, device_to_device_region);
  ASSERT_TRUE(status.ok()) << to_string(status);

  std::vector<int> out_values(32, -1);
  BufferTransferRegion device_to_host_region;
  device_to_host_region.size_bytes = buffer_desc.size_bytes;
  status = runtime->copy_to_host_async(
      consume_queue, out_values.data(), dst_buffer, device_to_host_region);
  ASSERT_TRUE(status.ok()) << to_string(status);
  status = runtime->synchronize_queue(consume_queue);
  ASSERT_TRUE(status.ok()) << to_string(status);

  EXPECT_EQ(out_values, in_values);
}

TEST(BackendRuntimeContractTest, NullQueueIsInvalidArgument) {
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
  const auto buffer_result = runtime->create_buffer(buffer_desc);
  ASSERT_TRUE(buffer_result.ok()) << to_string(buffer_result.error());
  const std::shared_ptr<BackendBuffer> buffer = buffer_result.value();

  const std::vector<float> values(8, 1.0f);
  BufferTransferRegion region;
  region.size_bytes = buffer_desc.size_bytes;
  BackendError status = runtime->copy_from_host_async(
      std::shared_ptr<BackendQueue>{}, buffer, values.data(), region);
  EXPECT_EQ(status.code, BackendErrorCode::InvalidArgument);
  EXPECT_FALSE(status.operation.empty());
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
      queue, dst_result.value(), src_result.value(), region);
  EXPECT_EQ(status.code, BackendErrorCode::InvalidArgument);
}

}  // namespace
