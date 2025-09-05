#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>

#include "cuda/common_host.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "utils/scope_timer.hpp"

namespace tinygs {

void GPUGaussian3d::copy_from_host(const Gaussian3d& gaussians) {
  TINYGS_TIMER("GPUGaussian3d::copy_from_host");
  const size_t num_gaussians = gaussians.means.size();

  // Resize device vectors
  m_means.resize(num_gaussians);
  m_opacities.resize(num_gaussians);
  m_rotations.resize(num_gaussians);
  m_scales.resize(num_gaussians);
  m_sh_coefficient_0.resize(num_gaussians);
  m_sh_coefficients_rest.resize(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1));

  // TODO: directly copy use cudaMemcpy if the input is already in pinned memory
  // Copy directly from SoA host vectors to device vectors
  thrust::copy(gaussians.means.begin(), gaussians.means.end(), m_means.begin());
  thrust::copy(gaussians.opacities.begin(), gaussians.opacities.end(), m_opacities.begin());
  thrust::copy(gaussians.rotations.begin(), gaussians.rotations.end(), m_rotations.begin());
  thrust::copy(gaussians.scales.begin(), gaussians.scales.end(), m_scales.begin());
  thrust::copy(gaussians.sh_coefficient_0.begin(), gaussians.sh_coefficient_0.end(), m_sh_coefficient_0.begin());
  thrust::copy(gaussians.sh_coefficients_rest.begin(), gaussians.sh_coefficients_rest.end(), m_sh_coefficients_rest.begin());
}

void GPUGaussian3d::copy_to_host(Gaussian3d& gaussians) {
  TINYGS_TIMER("GPUGaussian3d::copy_to_host");
  const size_t num_gaussians = m_means.size();

  // Resize host container vectors
  gaussians.means.resize(num_gaussians);
  gaussians.opacities.resize(num_gaussians);
  gaussians.rotations.resize(num_gaussians);
  gaussians.scales.resize(num_gaussians);
  gaussians.sh_coefficient_0.resize(num_gaussians);
  gaussians.sh_coefficients_rest.resize(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1));

  // Copy directly from device vectors to SoA host vectors
  thrust::copy(m_means.begin(), m_means.end(), gaussians.means.begin());
  thrust::copy(m_opacities.begin(), m_opacities.end(), gaussians.opacities.begin());
  thrust::copy(m_rotations.begin(), m_rotations.end(), gaussians.rotations.begin());
  thrust::copy(m_scales.begin(), m_scales.end(), gaussians.scales.begin());
  thrust::copy(m_sh_coefficient_0.begin(), m_sh_coefficient_0.end(), gaussians.sh_coefficient_0.begin());
  thrust::copy(m_sh_coefficients_rest.begin(), m_sh_coefficients_rest.end(), gaussians.sh_coefficients_rest.begin());
}

void GPUGaussian3d::memset_async(char value, cudaStream_t stream) {
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_means.data()), value, sizeof(float3) * m_means.size(), stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_opacities.data()), value, sizeof(float) * m_opacities.size(), stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_rotations.data()), value, sizeof(float4) * m_rotations.size(), stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_scales.data()), value, sizeof(float3) * m_scales.size(), stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_sh_coefficient_0.data()), value, sizeof(float3) * m_sh_coefficient_0.size(), stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_sh_coefficients_rest.data()), value, sizeof(float3) * m_sh_coefficients_rest.size(), stream));
}

void GPUGaussian3d::memset(char value) {
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_means.data()), value, sizeof(float3) * m_means.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_opacities.data()), value, sizeof(float) * m_opacities.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_rotations.data()), value, sizeof(float4) * m_rotations.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_scales.data()), value, sizeof(float3) * m_scales.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_sh_coefficient_0.data()), value, sizeof(float3) * m_sh_coefficient_0.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_sh_coefficients_rest.data()), value, sizeof(float3) * m_sh_coefficients_rest.size()));
}



__global__ void copy_gaussian_items(
  const vec3 * __restrict__ src_means,
  vec3 * __restrict__ dst_means,
  const float * __restrict__ src_opacities,
  float * __restrict__ dst_opacities,
  const vec4 * __restrict__ src_rotations,
  vec4 * __restrict__ dst_rotations,
  const vec3 * __restrict__ src_scales,
  vec3 * __restrict__ dst_scales,
  const vec3 * __restrict__ src_sh_coefficient_0,
  vec3 * __restrict__ dst_sh_coefficient_0,
  const vec3 * __restrict__ src_sh_coefficients_rest,
  vec3 * __restrict__ dst_sh_coefficients_rest,
  const int * __restrict__ mapping,
  int num_kept
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kept) return;

  int src_idx = mapping[idx];
  
  // Copy all fields
  dst_means[idx] = src_means[src_idx];
  dst_opacities[idx] = src_opacities[src_idx];
  dst_rotations[idx] = src_rotations[src_idx];
  dst_scales[idx] = src_scales[src_idx];
  dst_sh_coefficient_0[idx] = src_sh_coefficient_0[src_idx];
  
  // Copy rest SH coefficients
  int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
    dst_sh_coefficients_rest[dst_rest_start + i] = src_sh_coefficients_rest[src_rest_start + i];
  }
}

void GPUGaussian3d::remove(char* kept_flag, int num_kept) {
  size_t original_size = size();
  thrust::device_vector<int> mapping(original_size); // kept[idx] = original_idx

  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(original_size),
    mapping.begin(), [kept_flag] __device__ (int orig) { return static_cast<bool>(kept_flag[orig]); });

  // Create new vectors for all gaussian data
  thrust::device_vector<vec3> means(num_kept);
  thrust::device_vector<float> opacities(num_kept);
  thrust::device_vector<vec4> rotations(num_kept);
  thrust::device_vector<vec3> scales(num_kept);
  thrust::device_vector<vec3> sh_coefficient_0(num_kept);
  thrust::device_vector<vec3> sh_coefficients_rest(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));

  // Copy all items using the mapping
  const int grid = (num_kept + 255) / 256;
  copy_gaussian_items<<<grid, 256>>>(
      thrust::raw_pointer_cast(m_means.data()),
      thrust::raw_pointer_cast(means.data()),
      thrust::raw_pointer_cast(m_opacities.data()),
      thrust::raw_pointer_cast(opacities.data()),
      thrust::raw_pointer_cast(m_rotations.data()),
      thrust::raw_pointer_cast(rotations.data()),
      thrust::raw_pointer_cast(m_scales.data()),
      thrust::raw_pointer_cast(scales.data()),
      thrust::raw_pointer_cast(m_sh_coefficient_0.data()),
      thrust::raw_pointer_cast(sh_coefficient_0.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest.data()),
      thrust::raw_pointer_cast(mapping.data()),
      num_kept
  );

  // Move the new vectors to replace the old ones
  m_means = std::move(means);
  m_opacities = std::move(opacities);
  m_rotations = std::move(rotations);
  m_scales = std::move(scales);
  m_sh_coefficient_0 = std::move(sh_coefficient_0);
  m_sh_coefficients_rest = std::move(sh_coefficients_rest);
  
  CUDA_CHECK_THROW(cudaDeviceSynchronize()); CUDA_CHECK_THROW(cudaGetLastError());
}

void GPUGaussian3d::append(int num_dup) {
  assert(num_dup > 0);
  const size_t target_size = this->size() + static_cast<size_t>(num_dup);
  m_means.resize(target_size);
  m_opacities.resize(target_size);
  m_rotations.resize(target_size);
  m_scales.resize(target_size);
  m_sh_coefficient_0.resize(target_size);
  m_sh_coefficients_rest.resize(target_size * (kMaxSphericalHarmonicsCoefficients - 1));
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone_async(cudaStream_t stream) {
  auto gaussians = std::make_unique<GPUGaussian3d>();
  gaussians->m_means.resize(m_means.size());
  gaussians->m_opacities.resize(m_opacities.size());
  gaussians->m_rotations.resize(m_rotations.size());
  gaussians->m_scales.resize(m_scales.size());
  gaussians->m_sh_coefficient_0.resize(m_sh_coefficient_0.size());
  gaussians->m_sh_coefficients_rest.resize(m_sh_coefficients_rest.size());

  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_means.data()),
      thrust::raw_pointer_cast(m_means.data()),
      sizeof(float3) * m_means.size(),
      cudaMemcpyDeviceToDevice,
      stream));

  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_opacities.data()),
      thrust::raw_pointer_cast(m_opacities.data()),
      sizeof(float) * m_opacities.size(),
      cudaMemcpyDeviceToDevice,
      stream));

  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_rotations.data()),
      thrust::raw_pointer_cast(m_rotations.data()),
      sizeof(float4) * m_rotations.size(),
      cudaMemcpyDeviceToDevice,
      stream));

  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_scales.data()),
      thrust::raw_pointer_cast(m_scales.data()),
      sizeof(float3) * m_scales.size(),
      cudaMemcpyDeviceToDevice,
      stream));

  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_sh_coefficient_0.data()),
      thrust::raw_pointer_cast(m_sh_coefficient_0.data()),
      sizeof(float3) * m_sh_coefficient_0.size(),
      cudaMemcpyDeviceToDevice,
      stream));

  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_sh_coefficients_rest.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest.data()),
      sizeof(float3) * m_sh_coefficients_rest.size(),
      cudaMemcpyDeviceToDevice,
      stream));
  return gaussians;
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone() {
  auto gaussians = std::make_unique<GPUGaussian3d>();
  gaussians->m_means.resize(m_means.size());
  gaussians->m_opacities.resize(m_opacities.size());
  gaussians->m_rotations.resize(m_rotations.size());
  gaussians->m_scales.resize(m_scales.size());
  gaussians->m_sh_coefficient_0.resize(m_sh_coefficient_0.size());
  gaussians->m_sh_coefficients_rest.resize(m_sh_coefficients_rest.size());

  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_means.data()),
      thrust::raw_pointer_cast(m_means.data()),
      sizeof(float3) * m_means.size(),
      cudaMemcpyDeviceToDevice));

  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_opacities.data()),
      thrust::raw_pointer_cast(m_opacities.data()),
      sizeof(float) * m_opacities.size(),
      cudaMemcpyDeviceToDevice));

  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_rotations.data()),
      thrust::raw_pointer_cast(m_rotations.data()),
      sizeof(float4) * m_rotations.size(),
      cudaMemcpyDeviceToDevice));

  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_scales.data()),
      thrust::raw_pointer_cast(m_scales.data()),
      sizeof(float3) * m_scales.size(),
      cudaMemcpyDeviceToDevice));

  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_sh_coefficient_0.data()),
      thrust::raw_pointer_cast(m_sh_coefficient_0.data()),
      sizeof(float3) * m_sh_coefficient_0.size(),
      cudaMemcpyDeviceToDevice));

  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_sh_coefficients_rest.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest.data()),
      sizeof(float3) * m_sh_coefficients_rest.size(),
      cudaMemcpyDeviceToDevice));
  return gaussians;
}

}  // namespace tinygs