#include <thrust/execution_policy.h>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/sgd.hpp"

namespace tinygs {

__global__ void launch_gaussian_sgd_step_SoA(
  // Means
  vec3* __restrict__ means,
  const vec3* __restrict__ means_grad,
  // Opacities
  float* __restrict__ opacities,
  const float* __restrict__ opacities_grad,
  // Rotation
  vec4* __restrict__ rotations,
  const vec4* __restrict__ rotations_grad,
  // Scales
  vec3* __restrict__ scales,
  const vec3* __restrict__ scales_grad,
  // Spherical Harmonics
  vec3* __restrict__ sh_coefficient_0,
  const vec3* __restrict__ sh_coefficient_0_grad,
  vec3* __restrict__ sh_coefficients_rest,
  const vec3* __restrict__ sh_coefficients_rest_grad,
  // other
  GaussianOptimizationParams general_p,
  uint32_t num_gaussians,
  const float global_step_size
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians) return;
  const float inv_n = 1.0f / num_gaussians;

  // accumulate all the gradients, to cull out zero gradient gaussians
  const float grad_norm_1 = (
    sum(abs(means_grad[idx])) +
    abs(opacities_grad[idx]) +
    sum(abs(rotations_grad[idx])) +
    sum(abs(scales_grad[idx]))
  );
  if (grad_norm_1 == 0 && general_p.skip_zero_grad) return;

  // actually perform the optimization for this gaussian
  { // means
    vec3& val = means[idx];
    const vec3& grad = means_grad[idx];
    const vec3 grad_clipped = general_p.max_grad_1 != 0.0f ? 
      copysign(min(abs(grad), vec3(general_p.max_grad_1)), grad) : grad;
    val -= general_p.means_lr * global_step_size * grad_clipped;
  }

  { // opacities
    float& val = opacities[idx];
    float actual = logistic(val); // 0 < actual < 1 => L1(actual) = actual => dL1/dval = actual * (1 - actual)
    float grad = opacities_grad[idx] + (general_p.opacities_l1 * actual * (1 - actual))  / (float) num_gaussians;
    const float grad_clipped = general_p.max_grad_1 != 0.0f ? 
      copysign(min(abs(grad), general_p.max_grad_1), grad) : grad;
    val -= general_p.opacities_lr * global_step_size * grad_clipped;
  }

  { // rotations
    vec4& val = rotations[idx];
    const vec4& grad = rotations_grad[idx];
    const vec4 grad_clipped = general_p.max_grad_1 != 0.0f ? 
      copysign(min(abs(grad), vec4(general_p.max_grad_1)), grad) : grad;
    val -= general_p.rotations_lr * global_step_size * grad_clipped;
  }

  { // scales
    vec3& val = scales[idx];
    // actual > 0 => L1(actual) = actual => dL1/dval = actual
    vec3 grad = scales_grad[idx] + (general_p.scales_l1 * exp(val)) / (float) num_gaussians;
    const vec3 grad_clipped = general_p.max_grad_1 != 0.0f ? 
      copysign(min(abs(grad), vec3(general_p.max_grad_1)), grad) : grad;
    val -= general_p.scales_lr * global_step_size * grad_clipped;
  }

  { // spherical harmonics - 0th coefficient
    vec3& val = sh_coefficient_0[idx];
    const vec3& grad = sh_coefficient_0_grad[idx];
    const vec3 grad_clipped = general_p.max_grad_1 != 0.0f ? 
      copysign(min(abs(grad), vec3(general_p.max_grad_1)), grad) : grad;
    val -= general_p.shs_lr * global_step_size * grad_clipped;
  }

  { // spherical harmonics - rest coefficients
    // NOTE: They use 1/20 LR w.r.t. sh0
    int start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
    int end = start + (kMaxSphericalHarmonicsCoefficients - 1);
    for (int i = start; i < end; i++) {
      vec3& val = sh_coefficients_rest[i];
      const vec3& grad = sh_coefficients_rest_grad[i];
      const vec3 grad_clipped = general_p.max_grad_1 != 0.0f ? 
        copysign(min(abs(grad), vec3(general_p.max_grad_1)), grad) : grad;
      val -= general_p.shs_lr * 0.05f * global_step_size * grad_clipped;
    }
  }
}

void SGD::step(float scale) {
  const float global_step_size = scale;
  const int grid = (m_gaussians->size() + 255) / 256;

  launch_gaussian_sgd_step_SoA<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_gaussians->means().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
    thrust::raw_pointer_cast(m_gaussians->opacities().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
    thrust::raw_pointer_cast(m_gaussians->rotations().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
    thrust::raw_pointer_cast(m_gaussians->scales().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
    thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
    thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
    m_params,
    m_gaussians->size(),
    global_step_size
  );
  CUDA_CHECK_THROW(cudaDeviceSynchronize()); CUDA_CHECK_THROW(cudaGetLastError());
}

SGD::SGD(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad,
      const SGDParameters& params):
  OptimizerBase(gaussians, gaussians_grad), m_sgd_params(params) {
  // SGD doesn't need internal buffers like AdamW
}

void SGD::reset() {
  // SGD doesn't have internal state to reset
}

void SGD::remove(char* kept_flag, int num_kept) {
  // SGD doesn't have internal state to manage during removal
}

void SGD::duplicate(int* indices, int* new_indices, int num_duplicate) {
  // SGD doesn't have internal state to manage during duplication
  (void)indices;
  (void)new_indices;
  (void)num_duplicate;
}

void SGD::reset(int* indices, int num_reset) {
  // SGD doesn't have internal state to reset for specific gaussians
  (void)indices;
  (void)num_reset;
}

}  // namespace tinygs