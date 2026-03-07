#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "tinygs/core/gaussian.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/platform/runtime_factory.hpp"

namespace tinygs {
namespace {

Gaussian3d make_test_gaussians(int n) {
  Gaussian3d g;
  g.means.resize(static_cast<size_t>(n));
  g.opacities.resize(static_cast<size_t>(n));
  g.rotations.resize(static_cast<size_t>(n));
  g.scales.resize(static_cast<size_t>(n));
  g.sh0.resize(static_cast<size_t>(n) * 1ull);
  g.sh1.resize(static_cast<size_t>(n) * 3ull);
  g.sh2.resize(static_cast<size_t>(n) * 5ull);
  g.sh3.resize(static_cast<size_t>(n) * 7ull);
  for (int i = 0; i < n; ++i) {
    g.means[static_cast<size_t>(i)] = vec3(0.5f * i, -0.25f * i, 1.0f + 0.75f * i);
    g.opacities[static_cast<size_t>(i)] = -0.1f * static_cast<float>(i + 1);
    g.rotations[static_cast<size_t>(i)] = vec4(1.0f - 0.1f * i, 0.2f * i, -0.3f * i, 0.4f * i);
    g.scales[static_cast<size_t>(i)] = vec3(0.01f * (i + 1), 0.02f * (i + 1), 0.03f * (i + 1));
  }
  for (size_t i = 0; i < g.sh0.size(); ++i) g.sh0[i] = vec3(0.1f * i, 0.2f * i, 0.3f * i);
  for (size_t i = 0; i < g.sh1.size(); ++i) g.sh1[i] = vec3(-0.1f * i, 0.15f * i, 0.05f * i);
  for (size_t i = 0; i < g.sh2.size(); ++i) g.sh2[i] = vec3(0.07f * i, -0.03f * i, 0.09f * i);
  for (size_t i = 0; i < g.sh3.size(); ++i) g.sh3[i] = vec3(0.11f * i, 0.13f * i, -0.17f * i);
  return g;
}

void expect_equal_gaussians(const Gaussian3d& a, const Gaussian3d& b) {
  ASSERT_EQ(a.means.size(), b.means.size());
  ASSERT_EQ(a.opacities.size(), b.opacities.size());
  ASSERT_EQ(a.rotations.size(), b.rotations.size());
  ASSERT_EQ(a.scales.size(), b.scales.size());
  ASSERT_EQ(a.sh0.size(), b.sh0.size());
  ASSERT_EQ(a.sh1.size(), b.sh1.size());
  ASSERT_EQ(a.sh2.size(), b.sh2.size());
  ASSERT_EQ(a.sh3.size(), b.sh3.size());
  for (size_t i = 0; i < a.means.size(); ++i) {
    EXPECT_FLOAT_EQ(a.means[i].x, b.means[i].x);
    EXPECT_FLOAT_EQ(a.means[i].y, b.means[i].y);
    EXPECT_FLOAT_EQ(a.means[i].z, b.means[i].z);
    EXPECT_FLOAT_EQ(a.opacities[i], b.opacities[i]);
    EXPECT_FLOAT_EQ(a.rotations[i].x, b.rotations[i].x);
    EXPECT_FLOAT_EQ(a.rotations[i].y, b.rotations[i].y);
    EXPECT_FLOAT_EQ(a.rotations[i].z, b.rotations[i].z);
    EXPECT_FLOAT_EQ(a.rotations[i].w, b.rotations[i].w);
    EXPECT_FLOAT_EQ(a.scales[i].x, b.scales[i].x);
    EXPECT_FLOAT_EQ(a.scales[i].y, b.scales[i].y);
    EXPECT_FLOAT_EQ(a.scales[i].z, b.scales[i].z);
  }
  for (size_t i = 0; i < a.sh0.size(); ++i) {
    EXPECT_FLOAT_EQ(a.sh0[i].x, b.sh0[i].x);
    EXPECT_FLOAT_EQ(a.sh0[i].y, b.sh0[i].y);
    EXPECT_FLOAT_EQ(a.sh0[i].z, b.sh0[i].z);
  }
  for (size_t i = 0; i < a.sh1.size(); ++i) {
    EXPECT_FLOAT_EQ(a.sh1[i].x, b.sh1[i].x);
    EXPECT_FLOAT_EQ(a.sh1[i].y, b.sh1[i].y);
    EXPECT_FLOAT_EQ(a.sh1[i].z, b.sh1[i].z);
  }
  for (size_t i = 0; i < a.sh2.size(); ++i) {
    EXPECT_FLOAT_EQ(a.sh2[i].x, b.sh2[i].x);
    EXPECT_FLOAT_EQ(a.sh2[i].y, b.sh2[i].y);
    EXPECT_FLOAT_EQ(a.sh2[i].z, b.sh2[i].z);
  }
  for (size_t i = 0; i < a.sh3.size(); ++i) {
    EXPECT_FLOAT_EQ(a.sh3[i].x, b.sh3[i].x);
    EXPECT_FLOAT_EQ(a.sh3[i].y, b.sh3[i].y);
    EXPECT_FLOAT_EQ(a.sh3[i].z, b.sh3[i].z);
  }
}

class MetalGPUGaussianTest : public ::testing::Test {
protected:
  void SetUp() override {
    BackendConfig config{BackendType::Metal, 0};
    auto runtime_result = create_backend_runtime(config);
    ASSERT_TRUE(runtime_result.ok()) << runtime_result.error().message;
    runtime = runtime_result.value();
    auto queue_result = runtime->create_queue(QueueDesc{});
    ASSERT_TRUE(queue_result.ok()) << queue_result.error().message;
    queue = queue_result.value();
  }

  std::shared_ptr<BackendRuntime> runtime;
  std::shared_ptr<BackendQueue> queue;
};

TEST_F(MetalGPUGaussianTest, CopyRoundtripPreservesAllFields) {
  const Gaussian3d input = make_test_gaussians(4);
  GPUGaussian3d gpu(runtime);
  gpu.copy_from_host_async(input, queue);

  Gaussian3d output;
  gpu.copy_to_host_async(output, queue);
  auto sync_status = runtime->synchronize_queue(queue);
  ASSERT_TRUE(sync_status.ok()) << sync_status.message;

  expect_equal_gaussians(output, input);
}

TEST_F(MetalGPUGaussianTest, CloneAndCloneAsyncProduceDeepCopies) {
  const Gaussian3d input = make_test_gaussians(3);
  GPUGaussian3d gpu(runtime);
  gpu.copy_from_host(input, queue);

  auto cloned_sync = gpu.clone();
  auto cloned_async = gpu.clone_async();
  auto sync_status = runtime->synchronize_device();
  ASSERT_TRUE(sync_status.ok()) << sync_status.message;

  Gaussian3d out_sync;
  Gaussian3d out_async;
  cloned_sync->copy_to_host(out_sync, queue);
  cloned_async->copy_to_host(out_async, queue);
  expect_equal_gaussians(out_sync, input);
  expect_equal_gaussians(out_async, input);
}

TEST_F(MetalGPUGaussianTest, AppendRemoveAndReorderMaintainExpectedLayout) {
  const Gaussian3d input = make_test_gaussians(3);
  GPUGaussian3d gpu(runtime);
  gpu.copy_from_host(input, queue);

  gpu.append(2, queue);
  auto sync_status = runtime->synchronize_queue(queue);
  ASSERT_TRUE(sync_status.ok()) << sync_status.message;

  Gaussian3d after_append;
  gpu.copy_to_host(after_append, queue);
  ASSERT_EQ(after_append.means.size(), 5u);
  for (size_t i = 0; i < input.means.size(); ++i) {
    EXPECT_FLOAT_EQ(after_append.means[i].x, input.means[i].x);
    EXPECT_FLOAT_EQ(after_append.means[i].y, input.means[i].y);
    EXPECT_FLOAT_EQ(after_append.means[i].z, input.means[i].z);
    EXPECT_FLOAT_EQ(after_append.opacities[i], input.opacities[i]);
  }
  for (size_t i = input.means.size(); i < after_append.means.size(); ++i) {
    EXPECT_FLOAT_EQ(after_append.means[i].x, 0.0f);
    EXPECT_FLOAT_EQ(after_append.means[i].y, 0.0f);
    EXPECT_FLOAT_EQ(after_append.means[i].z, 0.0f);
    EXPECT_FLOAT_EQ(after_append.opacities[i], 0.0f);
  }

  auto kept_flag = create_device_buffer_for<char>(runtime, 5, "kept_flag");
  std::array<char, 5> keep = {1, 0, 1, 0, 1};
  copy_from_host(runtime, queue, kept_flag, keep.data(), keep.size());
  gpu.remove(buffer_data<char>(kept_flag), 3);

  auto reorder_idx = create_device_buffer_for<uint>(runtime, 3, "reorder_idx");
  std::array<uint, 3> reorder = {2u, 0u, 1u};
  copy_from_host(runtime, queue, reorder_idx, reorder.data(), reorder.size());
  gpu.reorder(buffer_data<uint>(reorder_idx));

  Gaussian3d output;
  gpu.copy_to_host(output, queue);
  ASSERT_EQ(output.means.size(), 3u);
  EXPECT_FLOAT_EQ(output.means[0].x, 0.0f);
  EXPECT_FLOAT_EQ(output.means[1].x, input.means[0].x);
  EXPECT_FLOAT_EQ(output.means[2].x, input.means[2].x);
}

TEST_F(MetalGPUGaussianTest, ComputeMortonIndicesAndDensificationReorderWork) {
  Gaussian3d input = make_test_gaussians(3);
  input.means[0] = vec3(0.0f, 0.0f, 0.0f);
  input.means[1] = vec3(0.5f, 0.5f, 0.5f);
  input.means[2] = vec3(1.0f, 1.0f, 1.0f);
  GPUGaussian3d gpu(runtime);
  gpu.copy_from_host(input, queue);

  auto morton = gpu.compute_morton_order_indices();
  ASSERT_NE(morton, nullptr);
  std::array<uint, 3> idx{};
  copy_to_host(runtime, queue, morton, idx.data(), idx.size());
  std::array<uint, 3> expected_idx = {0u, 1u, 2u};
  EXPECT_EQ(idx, expected_idx);

  std::vector<DensificationInfo> host_info(3);
  host_info[0].accum_counter = 10.0f;
  host_info[1].accum_counter = 20.0f;
  host_info[2].accum_counter = 30.0f;
  auto info = create_device_buffer_for<DensificationInfo>(runtime, host_info.size(), "info");
  copy_from_host(runtime, queue, info, host_info.data(), host_info.size());

  auto reordered = reorder_densification_info(info, idx.data(), idx.size(), runtime, nullptr);
  ASSERT_NE(reordered, nullptr);
  std::vector<DensificationInfo> reordered_host(idx.size());
  copy_to_host(runtime, queue, reordered, reordered_host.data(), reordered_host.size());
  ASSERT_EQ(reordered_host.size(), host_info.size());
  EXPECT_FLOAT_EQ(reordered_host[0].accum_counter, 10.0f);
  EXPECT_FLOAT_EQ(reordered_host[1].accum_counter, 20.0f);
  EXPECT_FLOAT_EQ(reordered_host[2].accum_counter, 30.0f);
}

TEST_F(MetalGPUGaussianTest, RemoveRejectsKeepingFromEmptySet) {
  GPUGaussian3d gpu(runtime);
  EXPECT_ANY_THROW(gpu.remove(nullptr, 1, queue.get()));
}

TEST_F(MetalGPUGaussianTest, QueueParameterPathsProduceConsistentResults) {
  const Gaussian3d input = make_test_gaussians(3);
  GPUGaussian3d gpu(runtime);
  gpu.copy_from_host(input, queue);

  gpu.memset_async(0, queue.get());
  auto sync_status = runtime->synchronize_queue(queue);
  ASSERT_TRUE(sync_status.ok()) << sync_status.message;

  Gaussian3d after_memset;
  gpu.copy_to_host(after_memset, queue);
  for (size_t i = 0; i < after_memset.opacities.size(); ++i) {
    EXPECT_FLOAT_EQ(after_memset.opacities[i], 0.0f);
  }

  gpu.copy_from_host(input, queue);
  auto cloned = gpu.clone_async(queue.get());
  ASSERT_NE(cloned, nullptr);
  Gaussian3d cloned_host;
  cloned->copy_to_host(cloned_host, queue);
  expect_equal_gaussians(cloned_host, input);

  auto morton = gpu.compute_morton_order_indices(queue.get());
  ASSERT_NE(morton, nullptr);
  std::array<uint, 3> idx{};
  copy_to_host(runtime, queue, morton, idx.data(), idx.size());

  auto reordered_info_src = create_device_buffer_for<DensificationInfo>(runtime, idx.size(), "info");
  std::array<DensificationInfo, 3> infos{};
  infos[0].accum_counter = 3.0f;
  infos[1].accum_counter = 7.0f;
  infos[2].accum_counter = 11.0f;
  copy_from_host(runtime, queue, reordered_info_src, infos.data(), infos.size());
  auto reordered_info = reorder_densification_info(
      reordered_info_src, idx.data(), idx.size(), runtime, queue.get());
  ASSERT_NE(reordered_info, nullptr);
}

}  // namespace
}  // namespace tinygs
