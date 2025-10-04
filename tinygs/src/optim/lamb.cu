#include <thrust/execution_policy.h>

#include <nvtx3/nvtx3.hpp>
#include <cooperative_groups.h>
#include <cmath>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/lamb.hpp"

namespace tinygs {

__device__ static __forceinline__ float lerp(float v0, float v1, float t) {
  return fmaf(t, v1, fmaf(-t, v0, v0));
}

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
    T out = activate_scale_deriv(theta);
    return out * regu_l1;
  }
  __forceinline__ __device__ float operator()(const float& theta) const noexcept {
    return regu_l1 * activate_scale_deriv(theta);
  }
};

template<typename DecayFunc = NoDecay>
__global__ void lamb_vec3(
  vec3* __restrict__ thetas,
  const vec3* __restrict__ thetas_grad,
  vec3* __restrict__ thetas_first,
  vec3* __restrict__ thetas_second,
  LambParameters lamb_p,
  float lr,
  uint32_t num_items,
  float gradient_scale,
  float bias_correction1,
  float bias_correction2_sqrt,
  float max_grad_1,
  DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  vec3 theta = thetas[idx];
  theta -= f(theta) * lr;

  vec3 g = thetas_grad[idx] * gradient_scale;
  if (max_grad_1 != 0.0f) {
    g = copysign(min(abs(g), vec3(max_grad_1)), g);
  }
  vec3 m = thetas_first[idx];
  vec3 v = thetas_second[idx];

  m.x = lerp(m.x, g.x, 1.0f - lamb_p.beta1);
  m.y = lerp(m.y, g.y, 1.0f - lamb_p.beta1);
  m.z = lerp(m.z, g.z, 1.0f - lamb_p.beta1);
  const vec3 g_sq = vec3(g.x * g.x, g.y * g.y, g.z * g.z);
  v.x = lerp(v.x, g_sq.x, 1.0f - lamb_p.beta2);
  v.y = lerp(v.y, g_sq.y, 1.0f - lamb_p.beta2);
  v.z = lerp(v.z, g_sq.z, 1.0f - lamb_p.beta2);

  vec3 m_hat = m / bias_correction1;
  vec3 denom = vec3(sqrtf(v.x) / bias_correction2_sqrt + lamb_p.epsilon,
                    sqrtf(v.y) / bias_correction2_sqrt + lamb_p.epsilon,
                    sqrtf(v.z) / bias_correction2_sqrt + lamb_p.epsilon);
  vec3 adam_step = m_hat / denom;

  const float theta_norm = sqrtf(theta.x * theta.x + theta.y * theta.y + theta.z * theta.z);
  const float step_norm = sqrtf(adam_step.x * adam_step.x + adam_step.y * adam_step.y + adam_step.z * adam_step.z);
  const float trust_ratio_raw = theta_norm / (step_norm + lamb_p.epsilon);
  const float trust_ratio = fminf(fmaxf(trust_ratio_raw, lamb_p.trust_ratio_min), lamb_p.trust_ratio_max);

  theta.x -= lr * trust_ratio * adam_step.x;
  theta.y -= lr * trust_ratio * adam_step.y;
  theta.z -= lr * trust_ratio * adam_step.z;

  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}

template<typename DecayFunc = NoDecay>
__global__ void lamb_vec4(
  vec4* __restrict__ thetas,
  const vec4* __restrict__ thetas_grad,
  vec4* __restrict__ thetas_first,
  vec4* __restrict__ thetas_second,
  LambParameters lamb_p,
  float lr,
  uint32_t num_items,
  float gradient_scale,
  float bias_correction1,
  float bias_correction2_sqrt,
  float max_grad_1,
  DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  vec4 theta = thetas[idx];
  theta -= f(theta) * lr;

  vec4 g = thetas_grad[idx] * gradient_scale;
  if (max_grad_1 != 0.0f) {
    g = copysign(min(abs(g), vec4(max_grad_1)), g);
  }
  vec4 m = thetas_first[idx];
  vec4 v = thetas_second[idx];

  m.x = lerp(m.x, g.x, 1.0f - lamb_p.beta1);
  m.y = lerp(m.y, g.y, 1.0f - lamb_p.beta1);
  m.z = lerp(m.z, g.z, 1.0f - lamb_p.beta1);
  m.w = lerp(m.w, g.w, 1.0f - lamb_p.beta1);
  const vec4 g_sq = vec4(g.x * g.x, g.y * g.y, g.z * g.z, g.w * g.w);
  v.x = lerp(v.x, g_sq.x, 1.0f - lamb_p.beta2);
  v.y = lerp(v.y, g_sq.y, 1.0f - lamb_p.beta2);
  v.z = lerp(v.z, g_sq.z, 1.0f - lamb_p.beta2);
  v.w = lerp(v.w, g_sq.w, 1.0f - lamb_p.beta2);

  vec4 m_hat = m / bias_correction1;
  vec4 denom = vec4(sqrtf(v.x) / bias_correction2_sqrt + lamb_p.epsilon,
                    sqrtf(v.y) / bias_correction2_sqrt + lamb_p.epsilon,
                    sqrtf(v.z) / bias_correction2_sqrt + lamb_p.epsilon,
                    sqrtf(v.w) / bias_correction2_sqrt + lamb_p.epsilon);
  vec4 adam_step = m_hat / denom;

  const float theta_norm = sqrtf(theta.x * theta.x + theta.y * theta.y + theta.z * theta.z + theta.w * theta.w);
  const float step_norm = sqrtf(adam_step.x * adam_step.x + adam_step.y * adam_step.y + adam_step.z * adam_step.z + adam_step.w * adam_step.w);
  const float trust_ratio_raw = theta_norm / (step_norm + lamb_p.epsilon);
  const float trust_ratio = fminf(fmaxf(trust_ratio_raw, lamb_p.trust_ratio_min), lamb_p.trust_ratio_max);

  theta.x -= lr * trust_ratio * adam_step.x;
  theta.y -= lr * trust_ratio * adam_step.y;
  theta.z -= lr * trust_ratio * adam_step.z;
  theta.w -= lr * trust_ratio * adam_step.w;

  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}

template<typename DecayFunc = NoDecay>
__global__ void lamb_float(
  float* __restrict__ thetas,
  const float* __restrict__ thetas_grad,
  float* __restrict__ thetas_first,
  float* __restrict__ thetas_second,
  LambParameters lamb_p,
  float lr,
  uint32_t num_items,
  float gradient_scale,
  float bias_correction1,
  float bias_correction2_sqrt,
  float max_grad_1,
  DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  float theta = thetas[idx];
  theta -= f(theta) * lr;

  float g = thetas_grad[idx] * gradient_scale;
  if (max_grad_1 != 0.0f) {
    g = copysignf(fminf(fabsf(g), max_grad_1), g);
  }
  float m = thetas_first[idx];
  float v = thetas_second[idx];

  m = lerp(m, g, 1.0f - lamb_p.beta1);
  const float g_sq = g * g;
  v = lerp(v, g_sq, 1.0f - lamb_p.beta2);

  const float m_hat = m / bias_correction1;
  const float denom = sqrtf(v) / bias_correction2_sqrt + lamb_p.epsilon;
  const float adam_step = m_hat / denom;

  const float trust_ratio_raw = fabsf(theta) / (fabsf(adam_step) + lamb_p.epsilon);
  const float trust_ratio = fminf(fmaxf(trust_ratio_raw, lamb_p.trust_ratio_min), lamb_p.trust_ratio_max);
  theta -= lr * trust_ratio * adam_step;

  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}

// Copy optimizer state for kept gaussians (used by remove)
__global__ static void copy_optimizer_state(
  const vec3* __restrict__ src_means_first,
  const vec3* __restrict__ src_means_second,
  vec3* __restrict__ dst_means_first,
  vec3* __restrict__ dst_means_second,
  const float* __restrict__ src_opacities_first,
  const float* __restrict__ src_opacities_second,
  float* __restrict__ dst_opacities_first,
  float* __restrict__ dst_opacities_second,
  const vec4* __restrict__ src_rotations_first,
  const vec4* __restrict__ src_rotations_second,
  vec4* __restrict__ dst_rotations_first,
  vec4* __restrict__ dst_rotations_second,
  const vec3* __restrict__ src_scales_first,
  const vec3* __restrict__ src_scales_second,
  vec3* __restrict__ dst_scales_first,
  vec3* __restrict__ dst_scales_second,
  const vec3* __restrict__ src_sh0_first,
  const vec3* __restrict__ src_sh0_second,
  vec3* __restrict__ dst_sh0_first,
  vec3* __restrict__ dst_sh0_second,
  const vec3* __restrict__ src_shrest_first,
  const vec3* __restrict__ src_shrest_second,
  vec3* __restrict__ dst_shrest_first,
  vec3* __restrict__ dst_shrest_second,
  const uint* __restrict__ mapping,
  int num_items
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  uint src_idx = mapping[idx];
  dst_means_first[idx] = src_means_first[src_idx];
  dst_means_second[idx] = src_means_second[src_idx];
  dst_opacities_first[idx] = src_opacities_first[src_idx];
  dst_opacities_second[idx] = src_opacities_second[src_idx];
  dst_rotations_first[idx] = src_rotations_first[src_idx];
  dst_rotations_second[idx] = src_rotations_second[src_idx];
  dst_scales_first[idx] = src_scales_first[src_idx];
  dst_scales_second[idx] = src_scales_second[src_idx];
  dst_sh0_first[idx] = src_sh0_first[src_idx];
  dst_sh0_second[idx] = src_sh0_second[src_idx];
  const int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  const int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < (int)(kMaxSphericalHarmonicsCoefficients - 1); i++) {
    dst_shrest_first[dst_rest_start + i] = src_shrest_first[src_rest_start + i];
    dst_shrest_second[dst_rest_start + i] = src_shrest_second[src_rest_start + i];
  }
}

// Reorder optimizer state based on indices (gather)
__global__ void reorder_optimizer_state(
  const vec3* __restrict__ src_means_first,
  const vec3* __restrict__ src_means_second,
  vec3* __restrict__ dst_means_first,
  vec3* __restrict__ dst_means_second,
  const float* __restrict__ src_opacities_first,
  const float* __restrict__ src_opacities_second,
  float* __restrict__ dst_opacities_first,
  float* __restrict__ dst_opacities_second,
  const vec4* __restrict__ src_rotations_first,
  const vec4* __restrict__ src_rotations_second,
  vec4* __restrict__ dst_rotations_first,
  vec4* __restrict__ dst_rotations_second,
  const vec3* __restrict__ src_scales_first,
  const vec3* __restrict__ src_scales_second,
  vec3* __restrict__ dst_scales_first,
  vec3* __restrict__ dst_scales_second,
  const vec3* __restrict__ src_sh0_first,
  const vec3* __restrict__ src_sh0_second,
  vec3* __restrict__ dst_sh0_first,
  vec3* __restrict__ dst_sh0_second,
  const vec3* __restrict__ src_shrest_first,
  const vec3* __restrict__ src_shrest_second,
  vec3* __restrict__ dst_shrest_first,
  vec3* __restrict__ dst_shrest_second,
  const uint* __restrict__ mapping,
  int num_items
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;
  uint src_idx = mapping[idx];
  dst_means_first[idx] = src_means_first[src_idx];
  dst_means_second[idx] = src_means_second[src_idx];
  dst_opacities_first[idx] = src_opacities_first[src_idx];
  dst_opacities_second[idx] = src_opacities_second[src_idx];
  dst_rotations_first[idx] = src_rotations_first[src_idx];
  dst_rotations_second[idx] = src_rotations_second[src_idx];
  dst_scales_first[idx] = src_scales_first[src_idx];
  dst_scales_second[idx] = src_scales_second[src_idx];
  dst_sh0_first[idx] = src_sh0_first[src_idx];
  dst_sh0_second[idx] = src_sh0_second[src_idx];
  const int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  const int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < (int)(kMaxSphericalHarmonicsCoefficients - 1); i++) {
    dst_shrest_first[dst_rest_start + i] = src_shrest_first[src_rest_start + i];
    dst_shrest_second[dst_rest_start + i] = src_shrest_second[src_rest_start + i];
  }
}

struct lamb_domain { static constexpr char const* name{"optim"}; };
using range = nvtx3::scoped_range_in<lamb_domain>;
using regstr = nvtx3::registered_string_in<lamb_domain>;
struct m_step { static constexpr char const* message{"lamb_step"}; };

void Lamb::step(float scale, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
  constexpr int block_size = 256;
  const float gradient_scale = scale;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("Lamb::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("Lamb::step: gaussians and gaussians_grad must have same size");
  }

  m_global_steps++;
  const float bias_correction1 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(m_lamb_params.beta1),
                     static_cast<double>(m_global_steps)));
  const float bias_correction2_sqrt = static_cast<float>(
      std::sqrt(1.0 - std::pow(static_cast<double>(m_lamb_params.beta2),
                               static_cast<double>(m_global_steps))));

  auto n = m_gaussians->size();
  const float scene_scale = m_gaussians->scene_scale();
  {
    auto msg = regstr::get<m_step>();
    nvtx3::event_attributes attr(msg, nvtx3::payload{n});
    range range(attr);

    // Means
    lamb_vec3<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->means().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
      thrust::raw_pointer_cast(m_means_first.data()),
      thrust::raw_pointer_cast(m_means_second.data()),
      m_lamb_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // Opacities
    lamb_float<OpacityDecay><<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      thrust::raw_pointer_cast(m_opacities_first.data()),
      thrust::raw_pointer_cast(m_opacities_second.data()),
      m_lamb_params,
      m_params.opacities_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      OpacityDecay(m_params.opacities_l1)
    );

    // Rotations
    lamb_vec4<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->rotations().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
      thrust::raw_pointer_cast(m_rotations_first.data()),
      thrust::raw_pointer_cast(m_rotations_second.data()),
      m_lamb_params,
      m_params.rotations_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // Scales
    lamb_vec3<ScaleDecay><<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->scales().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
      thrust::raw_pointer_cast(m_scales_first.data()),
      thrust::raw_pointer_cast(m_scales_second.data()),
      m_lamb_params,
      m_params.scales_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      ScaleDecay(m_params.scales_l1)
    );

    // SH coefficient 0
    lamb_vec3<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
      thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
      thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
      m_lamb_params,
      m_params.shs_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // SH coefficients rest
    const uint32_t num_rest = (kMaxSphericalHarmonicsCoefficients - 1);
    lamb_vec3<<<div_round_up<uint>(n * num_rest, block_size), block_size, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
      m_lamb_params,
      m_params.shs_lr * 0.05f * m_global_lr,
      n * num_rest,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    maybe_sync(stream);
  }
}

Lamb::Lamb(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad)
  : OptimizerBase(gaussians, gaussians_grad) {
  Lamb::reset();
}

void Lamb::reset() {
  size_t n = m_gaussians->size();
  m_means_first.resize(n, vec3(0.f));
  m_means_second.resize(n, vec3(0.f));
  m_opacities_first.resize(n, 0.f);
  m_opacities_second.resize(n, 0.f);
  m_rotations_first.resize(n, vec4(0.f));
  m_rotations_second.resize(n, vec4(0.f));
  m_scales_first.resize(n, vec3(0.f));
  m_scales_second.resize(n, vec3(0.f));
  m_sh_coefficient_0_first.resize(n, vec3(0.f));
  m_sh_coefficient_0_second.resize(n, vec3(0.f));
  m_sh_coefficients_rest_first.resize(n * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
  m_sh_coefficients_rest_second.resize(n * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
  m_global_steps = 0;
}

void Lamb::reset(int* indices, int num_reset) {
  thrust::for_each(
    thrust::device_ptr<int>(indices),
    thrust::device_ptr<int>(indices) + num_reset,
    [
      means_first = m_means_first.data(),
      means_second = m_means_second.data(),
      opacities_first = m_opacities_first.data(),
      opacities_second = m_opacities_second.data(),
      rotations_first = m_rotations_first.data(),
      rotations_second = m_rotations_second.data(),
      scales_first = m_scales_first.data(),
      scales_second = m_scales_second.data(),
      sh0_first = m_sh_coefficient_0_first.data(),
      sh0_second = m_sh_coefficient_0_second.data(),
      shrest_first = m_sh_coefficients_rest_first.data(),
      shrest_second = m_sh_coefficients_rest_second.data()
    ] __device__(int idx) {
      means_first[idx] = vec3(0.f);
      means_second[idx] = vec3(0.f);
      opacities_first[idx] = 0.f;
      opacities_second[idx] = 0.f;
      rotations_first[idx] = vec4(0.f);
      rotations_second[idx] = vec4(0.f);
      scales_first[idx] = vec3(0.f);
      scales_second[idx] = vec3(0.f);
      sh0_first[idx] = vec3(0.f);
      sh0_second[idx] = vec3(0.f);
      for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
        shrest_first[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
        shrest_second[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
      }
    }
  );
}

void Lamb::reset_opacity() {
  thrust::fill(m_opacities_first.begin(), m_opacities_first.end(), 0.f);
  thrust::fill(m_opacities_second.begin(), m_opacities_second.end(), 0.f);
}

void Lamb::remove(char* kept_flag, int num_kept) {
  size_t original_size = m_gaussians->size();
  thrust::device_vector<uint> mapping(original_size);
  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<uint>(0), thrust::make_counting_iterator<uint>(original_size),
    mapping.begin(), [kept_flag] __device__ (uint orig) { return static_cast<bool>(kept_flag[orig]); }
  );

  thrust::device_vector<vec3> means_first(num_kept), means_second(num_kept);
  thrust::device_vector<float> opacities_first(num_kept), opacities_second(num_kept);
  thrust::device_vector<vec4> rotations_first(num_kept), rotations_second(num_kept);
  thrust::device_vector<vec3> scales_first(num_kept), scales_second(num_kept);
  thrust::device_vector<vec3> sh0_first(num_kept), sh0_second(num_kept);
  thrust::device_vector<vec3> shrest_first(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));
  thrust::device_vector<vec3> shrest_second(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));

  const int grid = (num_kept + 255) / 256;
  copy_optimizer_state<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_means_first.data()),
    thrust::raw_pointer_cast(m_means_second.data()),
    thrust::raw_pointer_cast(means_first.data()),
    thrust::raw_pointer_cast(means_second.data()),
    thrust::raw_pointer_cast(m_opacities_first.data()),
    thrust::raw_pointer_cast(m_opacities_second.data()),
    thrust::raw_pointer_cast(opacities_first.data()),
    thrust::raw_pointer_cast(opacities_second.data()),
    thrust::raw_pointer_cast(m_rotations_first.data()),
    thrust::raw_pointer_cast(m_rotations_second.data()),
    thrust::raw_pointer_cast(rotations_first.data()),
    thrust::raw_pointer_cast(rotations_second.data()),
    thrust::raw_pointer_cast(m_scales_first.data()),
    thrust::raw_pointer_cast(m_scales_second.data()),
    thrust::raw_pointer_cast(scales_first.data()),
    thrust::raw_pointer_cast(scales_second.data()),
    thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
    thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
    thrust::raw_pointer_cast(sh0_first.data()),
    thrust::raw_pointer_cast(sh0_second.data()),
    thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
    thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
    thrust::raw_pointer_cast(shrest_first.data()),
    thrust::raw_pointer_cast(shrest_second.data()),
    thrust::raw_pointer_cast(mapping.data()),
    num_kept
  );

  m_means_first = std::move(means_first);
  m_means_second = std::move(means_second);
  m_opacities_first = std::move(opacities_first);
  m_opacities_second = std::move(opacities_second);
  m_rotations_first = std::move(rotations_first);
  m_rotations_second = std::move(rotations_second);
  m_scales_first = std::move(scales_first);
  m_scales_second = std::move(scales_second);
  m_sh_coefficient_0_first = std::move(sh0_first);
  m_sh_coefficient_0_second = std::move(sh0_second);
  m_sh_coefficients_rest_first = std::move(shrest_first);
  m_sh_coefficients_rest_second = std::move(shrest_second);
}

void Lamb::duplicate(int* /*indices*/, int* /*new_indices*/, int /*num_duplicate*/) {
  if (m_gaussians->size() == 0) return;
  const uint32_t n = m_gaussians->size();
  const uint32_t num_sh_rest = kMaxSphericalHarmonicsCoefficients - 1;
  m_means_first.resize(n, vec3(0.f));
  m_means_second.resize(n, vec3(0.f));
  m_opacities_first.resize(n, 0.f);
  m_opacities_second.resize(n, 0.f);
  m_rotations_first.resize(n, vec4(0.f));
  m_rotations_second.resize(n, vec4(0.f));
  m_scales_first.resize(n, vec3(0.f));
  m_scales_second.resize(n, vec3(0.f));
  m_sh_coefficient_0_first.resize(n, vec3(0.f));
  m_sh_coefficient_0_second.resize(n, vec3(0.f));
  m_sh_coefficients_rest_first.resize(n * num_sh_rest, vec3(0.f));
  m_sh_coefficients_rest_second.resize(n * num_sh_rest, vec3(0.f));
}

void Lamb::reorder(uint* indices) {
  int num_gaussians = m_means_first.size();
  thrust::device_vector<vec3> means_first(num_gaussians), means_second(num_gaussians);
  thrust::device_vector<float> opacities_first(num_gaussians), opacities_second(num_gaussians);
  thrust::device_vector<vec4> rotations_first(num_gaussians), rotations_second(num_gaussians);
  thrust::device_vector<vec3> scales_first(num_gaussians), scales_second(num_gaussians);
  thrust::device_vector<vec3> sh0_first(num_gaussians), sh0_second(num_gaussians);
  thrust::device_vector<vec3> shrest_first(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1));
  thrust::device_vector<vec3> shrest_second(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1));

  const int grid = (num_gaussians + 255) / 256;
  reorder_optimizer_state<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_means_first.data()),
    thrust::raw_pointer_cast(m_means_second.data()),
    thrust::raw_pointer_cast(means_first.data()),
    thrust::raw_pointer_cast(means_second.data()),
    thrust::raw_pointer_cast(m_opacities_first.data()),
    thrust::raw_pointer_cast(m_opacities_second.data()),
    thrust::raw_pointer_cast(opacities_first.data()),
    thrust::raw_pointer_cast(opacities_second.data()),
    thrust::raw_pointer_cast(m_rotations_first.data()),
    thrust::raw_pointer_cast(m_rotations_second.data()),
    thrust::raw_pointer_cast(rotations_first.data()),
    thrust::raw_pointer_cast(rotations_second.data()),
    thrust::raw_pointer_cast(m_scales_first.data()),
    thrust::raw_pointer_cast(m_scales_second.data()),
    thrust::raw_pointer_cast(scales_first.data()),
    thrust::raw_pointer_cast(scales_second.data()),
    thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
    thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
    thrust::raw_pointer_cast(sh0_first.data()),
    thrust::raw_pointer_cast(sh0_second.data()),
    thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
    thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
    thrust::raw_pointer_cast(shrest_first.data()),
    thrust::raw_pointer_cast(shrest_second.data()),
    indices,
    num_gaussians
  );

  m_means_first = std::move(means_first);
  m_means_second = std::move(means_second);
  m_opacities_first = std::move(opacities_first);
  m_opacities_second = std::move(opacities_second);
  m_rotations_first = std::move(rotations_first);
  m_rotations_second = std::move(rotations_second);
  m_scales_first = std::move(scales_first);
  m_scales_second = std::move(scales_second);
  m_sh_coefficient_0_first = std::move(sh0_first);
  m_sh_coefficient_0_second = std::move(sh0_second);
  m_sh_coefficients_rest_first = std::move(shrest_first);
  m_sh_coefficients_rest_second = std::move(shrest_second);
}

void Lamb::set_params(const json& config) {
  OptimizerBase::set_params(config);
  m_lamb_params.from_json(config);
}

json Lamb::get_params() const {
  json params = OptimizerBase::get_params();
  params["type"] = "lamb";
  json lp = m_lamb_params.to_json();
  for (auto& [key, value] : lp.items()) { params[key] = value; }
  return params;
}

json LambParameters::to_json() const {
  json j;
  j["beta1"] = beta1;
  j["beta2"] = beta2;
  j["epsilon"] = epsilon;
  j["trust_ratio_min"] = trust_ratio_min;
  j["trust_ratio_max"] = trust_ratio_max;
  return j;
}

void LambParameters::from_json(const json &config) {
  if (config.contains("beta1"))
    beta1 = config.at("beta1").get<float>();
  if (config.contains("beta2"))
    beta2 = config.at("beta2").get<float>();
  if (config.contains("epsilon"))
    epsilon = config.at("epsilon").get<float>();
  if (config.contains("trust_ratio_min"))
    trust_ratio_min = config.at("trust_ratio_min").get<float>();
  if (config.contains("trust_ratio_max"))
    trust_ratio_max = config.at("trust_ratio_max").get<float>();
}

} // namespace tinygs