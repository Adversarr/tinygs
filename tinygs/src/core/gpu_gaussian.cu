#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include "cuda/common_host.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
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