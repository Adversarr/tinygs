#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include "tinygs/core/gpu_gaussian.hpp"

namespace tinygs {

void GPUGaussian3d::copy_from_host(const Gaussian3d& gaussians) {
  const size_t num_gaussians = gaussians.means_opacities.size();

  // Resize device vectors
  m_means_opacities.resize(num_gaussians);
  m_rotations.resize(num_gaussians);
  m_scales.resize(num_gaussians);
  m_sh_coefficients.resize(num_gaussians * kMaxSphericalHarmonicsCoefficients);

  // Copy directly from SoA host vectors to device vectors
  thrust::copy(gaussians.means_opacities.begin(), gaussians.means_opacities.end(), m_means_opacities.begin());
  thrust::copy(gaussians.rotations.begin(), gaussians.rotations.end(), m_rotations.begin());
  thrust::copy(gaussians.scales.begin(), gaussians.scales.end(), m_scales.begin());
  thrust::copy(gaussians.sh_coefficients.begin(), gaussians.sh_coefficients.end(), m_sh_coefficients.begin());
}

void GPUGaussian3d::copy_to_host(Gaussian3d& gaussians) {
  const size_t num_gaussians = m_means_opacities.size();

  // Resize host container vectors
  gaussians.means_opacities.resize(num_gaussians);
  gaussians.rotations.resize(num_gaussians);
  gaussians.scales.resize(num_gaussians);
  gaussians.sh_coefficients.resize(num_gaussians * kMaxSphericalHarmonicsCoefficients);

  // Copy directly from device vectors to SoA host vectors
  thrust::copy(m_means_opacities.begin(), m_means_opacities.end(), gaussians.means_opacities.begin());
  thrust::copy(m_rotations.begin(), m_rotations.end(), gaussians.rotations.begin());
  thrust::copy(m_scales.begin(), m_scales.end(), gaussians.scales.begin());
  thrust::copy(m_sh_coefficients.begin(), m_sh_coefficients.end(), gaussians.sh_coefficients.begin());
}

}  // namespace tinygs