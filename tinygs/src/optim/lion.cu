#include <thrust/execution_policy.h>

#include <nvtx3/nvtx3.hpp>
#include <cooperative_groups.h>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/lion.hpp"

namespace tinygs {

// Simple helpers mirrored from SimpleAdam implementation
__device__ static __forceinline__ float lerp(float v0, float v1, float t) {
    return fmaf(t, v1, fmaf(-t, v0, v0));
}

// Regularization functors (L1 for opacities & scales)
struct NoDecay {
  __host__ __device__ NoDecay() {}
  template <typename T>
  __forceinline__ __device__ T operator()(const T& theta) const noexcept { return T(0.f); }
  __forceinline__ __device__ float operator()(const float& theta) const noexcept { return 0.f; }
};

struct OpacityDecay {
  float regu_l1;
  __host__ __device__ explicit OpacityDecay(float l1) : regu_l1(l1) {}
  __forceinline__ __device__ float operator()(const float& theta) const noexcept {
    return regu_l1 * activate_opacity_deriv(theta);
  }
};


struct ScaleDecay {
  float regu_l1;
  __host__ __device__ explicit ScaleDecay(float l1) : regu_l1(l1) {}
  template <typename T>
  __forceinline__ __device__ T operator()(const T& theta) const noexcept {
    T out;
    out.x = regu_l1 * activate_scale_deriv(theta.x);
    out.y = regu_l1 * activate_scale_deriv(theta.y);
    out.z = regu_l1 * activate_scale_deriv(theta.z);
    return out;
  }
  __forceinline__ __device__ float operator()(const float& theta) const noexcept {
    return regu_l1 * activate_scale_deriv(theta);
  }
};

// Elementwise Lion update on contiguous float buffers
template<typename DecayFunc = NoDecay>
__global__ void lion_kernel(
    float* __restrict__ thetas,
    const float* __restrict__ thetas_grad,
    float* __restrict__ thetas_momentum,
    LionParameters lion_p,
    float lr,
    uint32_t num_elements,
    float gradient_scale,
    float max_grad_1,
    DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;

  // Load
  float theta = thetas[idx];
  // Decoupled regularization (L1) similar to AdamW implementation
  theta -= lr * f(theta);

  float g = gradient_scale * thetas_grad[idx];
  if (max_grad_1 != 0.0f) {
    g = copysignf(fminf(fabsf(g), max_grad_1), g);
  }
  float m = thetas_momentum[idx];

  // Update direction u using evolved sign momentum rule
  const float u = lion_p.beta1 * m + (1.0f - lion_p.beta1) * g;
  const float sgn = (u > 0.0f) ? 1.0f : ((u < 0.0f) ? -1.0f : 0.0f);
  theta -= lr * sgn;

  // Momentum tracking (only first moment)
  m = lerp(m, g, 1.0f - lion_p.beta2);

  // Write back
  thetas[idx] = theta;
  thetas_momentum[idx] = m;
}

// Copy optimizer state for kept gaussians (used by remove)
__global__ void copy_optimizer_state(
  const vec3* __restrict__ src_means_m,
  vec3* __restrict__ dst_means_m,
  const float* __restrict__ src_opacities_m,
  float* __restrict__ dst_opacities_m,
  const vec4* __restrict__ src_rotations_m,
  vec4* __restrict__ dst_rotations_m,
  const vec3* __restrict__ src_scales_m,
  vec3* __restrict__ dst_scales_m,
  const vec3* __restrict__ src_sh0_m,
  vec3* __restrict__ dst_sh0_m,
  const vec3* __restrict__ src_shrest_m,
  vec3* __restrict__ dst_shrest_m,
  const uint* __restrict__ mapping,
  int num_items
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  uint src_idx = mapping[idx];
  dst_means_m[idx] = src_means_m[src_idx];
  dst_opacities_m[idx] = src_opacities_m[src_idx];
  dst_rotations_m[idx] = src_rotations_m[src_idx];
  dst_scales_m[idx] = src_scales_m[src_idx];
  dst_sh0_m[idx] = src_sh0_m[src_idx];
  // Copy SH rest
  const int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  const int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < (int)(kMaxSphericalHarmonicsCoefficients - 1); i++) {
    dst_shrest_m[dst_rest_start + i] = src_shrest_m[src_rest_start + i];
  }
}

// Reorder optimizer state based on indices (gather)
__global__ void reorder_optimizer_state(
  const vec3* __restrict__ src_means_m,
  vec3* __restrict__ dst_means_m,
  const float* __restrict__ src_opacities_m,
  float* __restrict__ dst_opacities_m,
  const vec4* __restrict__ src_rotations_m,
  vec4* __restrict__ dst_rotations_m,
  const vec3* __restrict__ src_scales_m,
  vec3* __restrict__ dst_scales_m,
  const vec3* __restrict__ src_sh0_m,
  vec3* __restrict__ dst_sh0_m,
  const vec3* __restrict__ src_shrest_m,
  vec3* __restrict__ dst_shrest_m,
  const uint* __restrict__ mapping,
  int num_items
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;
  uint src_idx = mapping[idx];
  dst_means_m[idx] = src_means_m[src_idx];
  dst_opacities_m[idx] = src_opacities_m[src_idx];
  dst_rotations_m[idx] = src_rotations_m[src_idx];
  dst_scales_m[idx] = src_scales_m[src_idx];
  dst_sh0_m[idx] = src_sh0_m[src_idx];
  // SH rest
  const int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  const int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < (int)(kMaxSphericalHarmonicsCoefficients - 1); i++) {
    dst_shrest_m[dst_rest_start + i] = src_shrest_m[src_rest_start + i];
  }
}

struct lion_domain { static constexpr char const* name{"optim"}; };
using range = nvtx3::scoped_range_in<lion_domain>;
using regstr = nvtx3::registered_string_in<lion_domain>;
struct m_step { static constexpr char const* message{"lion_step"}; };

void Lion::step(float scale, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
  const float gradient_scale = scale;
  constexpr int block_size = 256;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("Lion::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("Lion::step: gaussians and gaussians_grad must have same size");
  }

  auto n = m_gaussians->size();

  const float scene_scale = m_gaussians->scene_scale();
  {
    auto msg = regstr::get<m_step>();
    nvtx3::event_attributes attr(msg, nvtx3::payload{n});
    range range(attr);

    // Means (vec3)
    lion_kernel<<<div_round_up<uint>(n * 3, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->means().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
      (float*)thrust::raw_pointer_cast(m_means_m.data()),
      m_lion_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n * 3,
      gradient_scale,
      m_params.max_grad_1
    );

    // Opacities (float) with L1 decay
    lion_kernel<OpacityDecay><<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      (float*)thrust::raw_pointer_cast(m_opacities_m.data()),
      m_lion_params,
      m_params.opacities_lr * m_global_lr,
      n,
      gradient_scale,
      m_params.max_grad_1,
      OpacityDecay(m_params.opacities_l1)
    );

    // Rotations (vec4)
    lion_kernel<<<div_round_up<uint>(n * 4, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->rotations().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
      (float*)thrust::raw_pointer_cast(m_rotations_m.data()),
      m_lion_params,
      m_params.rotations_lr * m_global_lr,
      n * 4,
      gradient_scale,
      m_params.max_grad_1
    );

    // Scales (vec3) with L1 decay
    lion_kernel<ScaleDecay><<<div_round_up<uint>(n * 3, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->scales().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
      (float*)thrust::raw_pointer_cast(m_scales_m.data()),
      m_lion_params,
      m_params.scales_lr * m_global_lr,
      n * 3,
      gradient_scale,
      m_params.max_grad_1,
      ScaleDecay(m_params.scales_l1)
    );

    // SH coefficient 0 (vec3)
    lion_kernel<<<div_round_up<uint>(n * 3, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
      (float*)thrust::raw_pointer_cast(m_sh_coefficient_0_m.data()),
      m_lion_params,
      m_params.shs_lr * m_global_lr,
      n * 3,
      gradient_scale,
      m_params.max_grad_1
    );

    // SH coefficients rest (vec3 per coeff)
    const int sh_rest_size = n * (kMaxSphericalHarmonicsCoefficients - 1) * 3;
    lion_kernel<<<div_round_up<uint>(sh_rest_size, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
      (float*)thrust::raw_pointer_cast(m_sh_coefficients_rest_m.data()),
      m_lion_params,
      m_params.shs_lr * 0.05f * m_global_lr,
      sh_rest_size,
      gradient_scale,
      m_params.max_grad_1
    );

    maybe_sync(stream);
  }
}

Lion::Lion(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad)
  : OptimizerBase(gaussians, gaussians_grad) {
  Lion::reset();
}

void Lion::reset() {
  size_t num_gaussians = m_gaussians->size();
  m_means_m.resize(num_gaussians, vec3(0.f));
  m_opacities_m.resize(num_gaussians, 0.f);
  m_rotations_m.resize(num_gaussians, vec4(0.f));
  m_scales_m.resize(num_gaussians, vec3(0.f));
  m_sh_coefficient_0_m.resize(num_gaussians, vec3(0.f));
  m_sh_coefficients_rest_m.resize(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
}

void Lion::reset(int* indices, int num_reset) {
  thrust::for_each(
    thrust::device_ptr<int>(indices),
    thrust::device_ptr<int>(indices) + num_reset,
    [
      means_m = m_means_m.data(),
      opacities_m = m_opacities_m.data(),
      rotations_m = m_rotations_m.data(),
      scales_m = m_scales_m.data(),
      sh0_m = m_sh_coefficient_0_m.data(),
      shrest_m = m_sh_coefficients_rest_m.data()
    ] __device__(int idx) {
      means_m[idx] = vec3(0.f);
      opacities_m[idx] = 0.f;
      rotations_m[idx] = vec4(0.f);
      scales_m[idx] = vec3(0.f);
      sh0_m[idx] = vec3(0.f);
      for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
        shrest_m[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
      }
    }
  );
}

void Lion::reset_opacity() {
  thrust::fill(m_opacities_m.begin(), m_opacities_m.end(), 0.f);
}

void Lion::remove(char* kept_flag, int num_kept) {
  size_t original_size = m_gaussians->size();
  thrust::device_vector<uint> mapping(original_size);
  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<uint>(0), thrust::make_counting_iterator<uint>(original_size),
    mapping.begin(), [kept_flag] __device__ (uint orig) { return static_cast<bool>(kept_flag[orig]); }
  );

  // Allocate destination buffers
  thrust::device_vector<vec3> means_m(num_kept);
  thrust::device_vector<float> opacities_m(num_kept);
  thrust::device_vector<vec4> rotations_m(num_kept);
  thrust::device_vector<vec3> scales_m(num_kept);
  thrust::device_vector<vec3> sh0_m(num_kept);
  thrust::device_vector<vec3> shrest_m(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));

  const int grid = (num_kept + 255) / 256;
  copy_optimizer_state<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_means_m.data()),
    thrust::raw_pointer_cast(means_m.data()),
    thrust::raw_pointer_cast(m_opacities_m.data()),
    thrust::raw_pointer_cast(opacities_m.data()),
    thrust::raw_pointer_cast(m_rotations_m.data()),
    thrust::raw_pointer_cast(rotations_m.data()),
    thrust::raw_pointer_cast(m_scales_m.data()),
    thrust::raw_pointer_cast(scales_m.data()),
    thrust::raw_pointer_cast(m_sh_coefficient_0_m.data()),
    thrust::raw_pointer_cast(sh0_m.data()),
    thrust::raw_pointer_cast(m_sh_coefficients_rest_m.data()),
    thrust::raw_pointer_cast(shrest_m.data()),
    thrust::raw_pointer_cast(mapping.data()),
    num_kept
  );

  m_means_m = std::move(means_m);
  m_opacities_m = std::move(opacities_m);
  m_rotations_m = std::move(rotations_m);
  m_scales_m = std::move(scales_m);
  m_sh_coefficient_0_m = std::move(sh0_m);
  m_sh_coefficients_rest_m = std::move(shrest_m);
}

void Lion::duplicate(int* /*indices*/, int* /*new_indices*/, int /*num_duplicate*/) {
  if (m_gaussians->size() == 0) return;
  const uint32_t n = m_gaussians->size();
  const uint32_t num_sh_rest = kMaxSphericalHarmonicsCoefficients - 1;
  m_means_m.resize(n, vec3(0.f));
  m_opacities_m.resize(n, 0.f);
  m_rotations_m.resize(n, vec4(0.f));
  m_scales_m.resize(n, vec3(0.f));
  m_sh_coefficient_0_m.resize(n, vec3(0.f));
  m_sh_coefficients_rest_m.resize(n * num_sh_rest, vec3(0.f));
}

void Lion::reorder(uint* indices) {
  int num_gaussians = m_means_m.size();
  thrust::device_vector<vec3> means_m(num_gaussians);
  thrust::device_vector<float> opacities_m(num_gaussians);
  thrust::device_vector<vec4> rotations_m(num_gaussians);
  thrust::device_vector<vec3> scales_m(num_gaussians);
  thrust::device_vector<vec3> sh0_m(num_gaussians);
  thrust::device_vector<vec3> shrest_m(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1));

  const int grid = (num_gaussians + 255) / 256;
  reorder_optimizer_state<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_means_m.data()),
    thrust::raw_pointer_cast(means_m.data()),
    thrust::raw_pointer_cast(m_opacities_m.data()),
    thrust::raw_pointer_cast(opacities_m.data()),
    thrust::raw_pointer_cast(m_rotations_m.data()),
    thrust::raw_pointer_cast(rotations_m.data()),
    thrust::raw_pointer_cast(m_scales_m.data()),
    thrust::raw_pointer_cast(scales_m.data()),
    thrust::raw_pointer_cast(m_sh_coefficient_0_m.data()),
    thrust::raw_pointer_cast(sh0_m.data()),
    thrust::raw_pointer_cast(m_sh_coefficients_rest_m.data()),
    thrust::raw_pointer_cast(shrest_m.data()),
    indices,
    num_gaussians
  );

  m_means_m = std::move(means_m);
  m_opacities_m = std::move(opacities_m);
  m_rotations_m = std::move(rotations_m);
  m_scales_m = std::move(scales_m);
  m_sh_coefficient_0_m = std::move(sh0_m);
  m_sh_coefficients_rest_m = std::move(shrest_m);
}

void Lion::set_params(const json& config) {
  OptimizerBase::set_params(config);
  m_lion_params.from_json(config);
}

json Lion::get_params() const {
  json params = OptimizerBase::get_params();
  params["type"] = "lion";
  json lp = m_lion_params.to_json();
  for (auto& [key, value] : lp.items()) { params[key] = value; }
  return params;
}

} // namespace tinygs