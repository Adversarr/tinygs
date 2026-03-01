#include <cuda/pipeline>
#include <thrust/execution_policy.h>
#include <cuda/barrier>
#include <nvtx3/nvtx3.hpp>
#include <cooperative_groups.h>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adam.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cstdio>

#include "../helper_math.h"

namespace cg = cooperative_groups;

// constexpr float kShRestScale = 1;
constexpr float kShRestScale = 0.05f;
constexpr int block_size = 512; // make occupancy higher

namespace tinygs {

__device__ static __forceinline__ float lerp(float v0, float v1, float t) {
    return fmaf(t, v1, fmaf(-t, v0, v0));
}

struct NoDecay {
  template <typename T>
  __forceinline__ __device__ auto operator()(const T& /* theta */, const T &g) const noexcept {
    return g;
  }

  template <typename T>
  __forceinline__ __device__ auto operator()(const T& /* theta */) const noexcept {
    return T(0.f);
  }

  __forceinline__ __device__ float4 operator()(const float4& /* theta */) const noexcept {
    return make_float4(0.f, 0.f, 0.f, 0.f);
  }
};

struct OpacityDecay {
  float regu_l1 = 0.f;

  template <typename T>
  __forceinline__ __device__ auto operator()(const T& theta, const T &g) const noexcept {
    return g + regu_l1 * activate_opacity_deriv(theta);
  }

  __forceinline__ __device__ float4 operator()(const float4& theta, const float4 &g) const noexcept {
    return make_float4(
        g.x + regu_l1 * activate_opacity_deriv(theta.x),
        g.y + regu_l1 * activate_opacity_deriv(theta.y),
        g.z + regu_l1 * activate_opacity_deriv(theta.z),
        g.w + regu_l1 * activate_opacity_deriv(theta.w));
  }

  template <typename T>
  __forceinline__ __device__ auto operator()(const T& theta) const noexcept {
    return regu_l1 * activate_opacity_deriv(theta);
  }

  __forceinline__ __device__ float4 operator()(const float4& theta) const noexcept {
    return make_float4(
        regu_l1 * activate_opacity_deriv(theta.x),
        regu_l1 * activate_opacity_deriv(theta.y),
        regu_l1 * activate_opacity_deriv(theta.z),
        regu_l1 * activate_opacity_deriv(theta.w));
  }
};

struct ScaleDecay {
  float regu_l1 = 0.f;

  template <typename T>
  __forceinline__ __device__ auto operator()(const T& theta, const T &g) const noexcept {
    return g + regu_l1 * activate_scale_deriv(theta);
  }

  __forceinline__ __device__ float4 operator()(const float4& theta, const float4 &g) const noexcept {
    return make_float4(
        g.x + regu_l1 * activate_scale_deriv(theta.x),
        g.y + regu_l1 * activate_scale_deriv(theta.y),
        g.z + regu_l1 * activate_scale_deriv(theta.z),
        g.w + regu_l1 * activate_scale_deriv(theta.w));
  }

  template <typename T>
  __forceinline__ __device__ auto operator()(const T& theta) const noexcept {
    return regu_l1 * activate_scale_deriv(theta);
  }

  __forceinline__ __device__ float4 operator()(const float4& theta) const noexcept {
    return make_float4(
        regu_l1 * activate_scale_deriv(theta.x),
        regu_l1 * activate_scale_deriv(theta.y),
        regu_l1 * activate_scale_deriv(theta.z),
        regu_l1 * activate_scale_deriv(theta.w));
  }
};

template<typename DecayFunc = NoDecay>
__global__ static void adam(
    // Means
    float *__restrict__ thetas,
    float const *__restrict__ thetas_grad,
    float *__restrict__ thetas_first,
    float *__restrict__ thetas_second,
    // other
    AdamParameters adam_p,
    float lr,
    uint32_t num_gaussians,
    float gradient_scale,
    float bias_correction1,     // (1 - beta_1^t)
    float bias_correction2_sqrt, // sqrt(1 - beta_2^t)
    float max_grad_1,
    DecayFunc f = DecayFunc()
) {
  const auto block_first = blockIdx.x * blockDim.x;
  const auto local_idx = threadIdx.x;
  const auto idx = block_first + local_idx;
  auto block = cg::this_thread_block();
  if (idx >= num_gaussians) return;

  // Load
  float theta = thetas[idx];
  float g = f(theta, gradient_scale * thetas_grad[idx]);
  if (max_grad_1 != 0.0f) {
    g = copysignf(fminf(fabsf(g), max_grad_1), g);
  }
  float m = thetas_first[idx];
  float v = thetas_second[idx];
  const float g_sq = g * g;

  // Update biased first and second moment estimates
  m = lerp(m, g, 1.0f - adam_p.beta1);
  v = lerp(v, g_sq, 1.0f - adam_p.beta2);

  // Bias-corrected estimates
  const float m_hat = m / bias_correction1;
  const float denom = tinygs::sqrt(v) / bias_correction2_sqrt + adam_p.epsilon;

  // step
  theta -= m_hat * lr / denom;

  // write back updated params and moments
  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}


template<typename DecayFunc = NoDecay>
__global__ static void adam_f32x4(
    // Means
    float4 *__restrict__ thetas,
    float4 const *__restrict__ thetas_grad,
    float4 *__restrict__ thetas_first,
    float4 *__restrict__ thetas_second,
    // other
    AdamParameters adam_p,
    float lr,
    uint32_t num_gaussians,
    float gradient_scale,
    float bias_correction1,     // (1 - beta_1^t)
    float bias_correction2_sqrt, // sqrt(1 - beta_2^t)
    float max_grad_1,
    DecayFunc f = DecayFunc()
) {
  const auto block_first = blockIdx.x * blockDim.x;
  const auto local_idx = threadIdx.x;
  const auto idx = block_first + local_idx;
  auto block = cg::this_thread_block();
  if (idx * 4 >= num_gaussians) return;

  // Load
  float4 theta = thetas[idx];
  const float4 g_raw = thetas_grad[idx];
  float4 g = f(theta, gradient_scale * g_raw);

  if (max_grad_1 != 0.0f) {
    g = make_float4(
        copysignf(fminf(fabsf(g.x), max_grad_1), g.x),
        copysignf(fminf(fabsf(g.y), max_grad_1), g.y),
        copysignf(fminf(fabsf(g.z), max_grad_1), g.z),
        copysignf(fminf(fabsf(g.w), max_grad_1), g.w));
  }
  float4 m = thetas_first[idx];
  float4 v = thetas_second[idx];
  const float4 g_sq = g * g;

  // Update biased first and second moment estimates
  m = lerp(m, g, 1.0f - adam_p.beta1);
  v = lerp(v, g_sq, 1.0f - adam_p.beta2);

  // Bias-corrected estimates
  const float4 m_hat = m / bias_correction1;
  const float4 denom = make_float4(
      sqrt(v.x) / bias_correction2_sqrt + adam_p.epsilon,
      sqrt(v.y) / bias_correction2_sqrt + adam_p.epsilon,
      sqrt(v.z) / bias_correction2_sqrt + adam_p.epsilon,
      sqrt(v.w) / bias_correction2_sqrt + adam_p.epsilon);

  // step
  theta -= m_hat * lr / denom;

  // write back updated params and moments
  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}


template<typename DecayFunc = NoDecay>
__global__ static void adamw_f32x4(
    float4 *__restrict__ thetas,
    float4 const *__restrict__ thetas_grad,
    float4 *__restrict__ thetas_first,
    float4 *__restrict__ thetas_second,
    AdamParameters adam_p,
    float lr,
    uint32_t num_gaussians,
    float gradient_scale,
    float bias_correction1,
    float bias_correction2_sqrt,
    float max_grad_1,
    DecayFunc f = DecayFunc()
) {
  const auto block_first = blockIdx.x * blockDim.x;
  const auto local_idx = threadIdx.x;
  const auto idx = block_first + local_idx;
  auto block = cg::this_thread_block();
  if (idx * 4 >= num_gaussians) return;

  // Load
  float4 theta = thetas[idx];
  // Decoupled weight decay
  float4 decay = f(theta);
  theta = make_float4(
      theta.x - lr * decay.x,
      theta.y - lr * decay.y,
      theta.z - lr * decay.z,
      theta.w - lr * decay.w);

  const float4 g_raw = thetas_grad[idx];
  float4 g = make_float4(
      gradient_scale * g_raw.x,
      gradient_scale * g_raw.y,
      gradient_scale * g_raw.z,
      gradient_scale * g_raw.w);

  if (max_grad_1 != 0.0f) {
    g = make_float4(
        copysignf(fminf(fabsf(g.x), max_grad_1), g.x),
        copysignf(fminf(fabsf(g.y), max_grad_1), g.y),
        copysignf(fminf(fabsf(g.z), max_grad_1), g.z),
        copysignf(fminf(fabsf(g.w), max_grad_1), g.w));
  }
  float4 m = thetas_first[idx];
  float4 v = thetas_second[idx];
  const float4 g_sq = make_float4(g.x * g.x, g.y * g.y, g.z * g.z, g.w * g.w);

  // Update biased first and second moment estimates
  m = lerp(m, g, 1.0f - adam_p.beta1);
  v = lerp(v, g_sq, 1.0f - adam_p.beta2);

  // Bias-corrected estimates
  const float4 m_hat = make_float4(
      m.x / bias_correction1,
      m.y / bias_correction1,
      m.z / bias_correction1,
      m.w / bias_correction1);
  const float4 denom = make_float4(
      sqrt(v.x) / bias_correction2_sqrt + adam_p.epsilon,
      sqrt(v.y) / bias_correction2_sqrt + adam_p.epsilon,
      sqrt(v.z) / bias_correction2_sqrt + adam_p.epsilon,
      sqrt(v.w) / bias_correction2_sqrt + adam_p.epsilon);

  // step
  theta = make_float4(
      theta.x - (m_hat.x * lr) / denom.x,
      theta.y - (m_hat.y * lr) / denom.y,
      theta.z - (m_hat.z * lr) / denom.z,
      theta.w - (m_hat.w * lr) / denom.w);

  // write back updated params and moments
  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}


template<typename DecayFunc = NoDecay>
__global__ static void adamw(
    // Means
    float *__restrict__ thetas,
    float const *__restrict__ thetas_grad,
    float *__restrict__ thetas_first,
    float *__restrict__ thetas_second,
    // other
    AdamParameters adam_p,
    float lr,
    uint32_t num_gaussians,
    float gradient_scale,
    float bias_correction1,     // (1 - beta_1^t)
    float bias_correction2_sqrt, // sqrt(1 - beta_2^t)
    float max_grad_1,
    DecayFunc f = DecayFunc()
) {

  const auto block_first = blockIdx.x * blockDim.x;
  const auto local_idx = threadIdx.x;
  const auto idx = block_first + local_idx;
  auto block = cg::this_thread_block();
  if (idx >= num_gaussians) return;

  // Load
  float theta = thetas[idx];
  theta -= lr * f(theta);
  float g = gradient_scale * thetas_grad[idx];
  if (max_grad_1 != 0.0f) {
    g = copysignf(fminf(fabsf(g), max_grad_1), g);
  }
  float m = thetas_first[idx];
  float v = thetas_second[idx];
  const float g_sq = g * g;

  // Update biased first and second moment estimates
  m = lerp(m, g, 1.0f - adam_p.beta1);
  v = lerp(v, g_sq, 1.0f - adam_p.beta2);

  // Bias-corrected estimates
  const float m_hat = m / bias_correction1;
  const float denom = tinygs::sqrt(v) / bias_correction2_sqrt + adam_p.epsilon;

  // step
  theta -= m_hat * lr / denom;

  // write back updated params and moments
  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}

struct Adam_domain {
  static constexpr char const *name{"optim"};
};
using range = nvtx3::scoped_range_in<Adam_domain>;
using attr = nvtx3::event_attributes;
using regstr = nvtx3::registered_string_in<Adam_domain>;
using ncat = nvtx3::named_category_in<Adam_domain>;
struct m_step {
  static constexpr char const *message{"adam_step"};
};

namespace {
// Configurable (via JSON) logging intervals
int g_momentum_log_interval = 1000;
int g_gradient_log_interval = 100;

// Functors for L1 accumulation
struct Vec3AbsSum {
  __host__ __device__ float operator()(const vec3& v) const {
    return fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
  }
};
struct Vec4AbsSum {
  __host__ __device__ float operator()(const vec4& v) const {
    return fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
  }
};
struct FloatAbs {
  __host__ __device__ float operator()(float v) const {
    return fabsf(v);
  }
};

// Simplified helpers (use thrust::device)
template<typename It, typename UnaryOp>
float l1_norm(It begin, It end, UnaryOp op) {
  if (begin == end) return 0.f;
  return thrust::transform_reduce(thrust::device, begin, end, op, 0.f, thrust::plus<float>());
}

template <typename VecT, typename It>
float l0_norm(It begin, It end) {
  if (begin == end) return 0.f;
  return thrust::transform_reduce(
      thrust::device, begin, end,
      [] __device__(VecT v) -> float {
        float l = glm::length(v);
        return l != 0.f ? 1.f : 0.f;
      },
      0.f, thrust::plus<float>());
}

template<typename VecT, typename AbsFunctor>
float l1_vec(const thrust::device_vector<VecT>& v, AbsFunctor f) {
  if (v.empty()) return 0.f;
  return l1_norm(v.begin(), v.end(), f) / (l0_norm<VecT>(v.begin(), v.end()) + FLT_EPSILON);
}
} // anonymous namespace

void Adam::step(float scale, cudaStream_t stream) {
  if (m_adam_params.decouple_decay) {
    step_adamw(scale, stream);
  } else {
    step_adam(scale, stream);
  }

  return;
  // Logging (performed after update; uses same stream for ordering)
  if (m_global_steps % g_momentum_log_interval == 0 || m_global_steps % g_gradient_log_interval == 0) {
    // Momentum L1
    float m_means_l1 = 0.f, m_opacities_l1 = 0.f, m_rot_l1 = 0.f, m_scales_l1 = 0.f, m_sh0_l1 = 0.f, m_shrest_l1 = 0.f;
    if (m_global_steps % g_momentum_log_interval == 0) {
      m_means_l1     = l1_vec(m_means_first, Vec3AbsSum{});
      m_opacities_l1 = l1_vec(m_opacities_first, FloatAbs{});
      m_rot_l1       = l1_vec(m_rotations_first, Vec4AbsSum{});
      m_scales_l1    = l1_vec(m_scales_first, Vec3AbsSum{});
      m_sh0_l1       = l1_vec(m_sh_coefficient_0_first, Vec3AbsSum{});
      m_shrest_l1    = l1_vec(m_sh_coefficients_rest_first, Vec3AbsSum{});
    }

    // Gradient L1
    float g_means_l1 = 0.f, g_opacities_l1 = 0.f, g_rot_l1 = 0.f, g_scales_l1 = 0.f, g_sh0_l1 = 0.f, g_shrest_l1 = 0.f;
    if (m_global_steps % g_gradient_log_interval == 0) {
      g_means_l1     = l1_vec(m_gaussians_grad->means(), Vec3AbsSum{});
      g_opacities_l1 = l1_vec(m_gaussians_grad->opacities(), FloatAbs{});
      g_rot_l1       = l1_vec(m_gaussians_grad->rotations(), Vec4AbsSum{});
      g_scales_l1    = l1_vec(m_gaussians_grad->scales(), Vec3AbsSum{});
      g_sh0_l1       = l1_vec(m_gaussians_grad->sh_coefficient_0(), Vec3AbsSum{});
      g_shrest_l1    = l1_vec(m_gaussians_grad->sh_coefficients_rest(), Vec3AbsSum{});
    }

    // Ensure reductions complete before host printf
    cudaStreamSynchronize(stream);

    if (m_global_steps % g_momentum_log_interval == 0) {
      std::printf("[Adam][step %llu] Momentum L1 | means=%.4g opacities=%.4g rotations=%.4g scales=%.4g sh0=%.4g shRest=%.4g total=%.4g\n",
                  (unsigned long long)m_global_steps,
                  m_means_l1, m_opacities_l1, m_rot_l1, m_scales_l1, m_sh0_l1, m_shrest_l1,
                  m_means_l1 + m_opacities_l1 + m_rot_l1 + m_scales_l1 + m_sh0_l1 + m_shrest_l1);
    }
    if (m_global_steps % g_gradient_log_interval == 0) {
      std::printf("[Adam][step %llu] Grad L1     | means=%.4g opacities=%.4g rotations=%.4g scales=%.4g sh0=%.4g shRest=%.4g total=%.4g\n",
                  (unsigned long long)m_global_steps,
                  g_means_l1, g_opacities_l1, g_rot_l1, g_scales_l1, g_sh0_l1, g_shrest_l1,
                  g_means_l1 + g_opacities_l1 + g_rot_l1 + g_scales_l1 + g_sh0_l1 + g_shrest_l1);
    }
  }
}


void Adam::step_adam(float scale, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
  const float gradient_scale = scale;  // This is the gradient scaler, not learning rate multiplier
  constexpr int block_size = 256;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("Adam::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("Adam::step: gaussians and gaussians_grad must have same size");
  }

  auto n = m_gaussians->size();
  float g_scale = 1.0f;
  if (m_adam_params.decay_reduction == "mean") {
    g_scale = 1.0f / n;
  }

  m_global_steps++;
  const float bias_correction1 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                     static_cast<double>(m_global_steps)));
  const float bias_correction2_sqrt = static_cast<float>(
      std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                               static_cast<double>(m_global_steps))));

  const float scene_scale = m_gaussians->scene_scale();
  {
    auto msg = regstr::get<m_step>();
    nvtx3::event_attributes attr(msg, nvtx3::payload{n});
    range range(attr);
    adam_f32x4<<<div_round_up<uint>((n * 3 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->means().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
      (float4*) thrust::raw_pointer_cast(m_means_first.data()),
      (float4*) thrust::raw_pointer_cast(m_means_second.data()),
      m_adam_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // Opacities
    adam<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_opacities_first.data()),
      (float*) thrust::raw_pointer_cast(m_opacities_second.data()),
      m_adam_params,
      m_params.opacities_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      OpacityDecay(m_params.opacities_l1 * g_scale)
    );

    // Rotations
    adam_f32x4<<<div_round_up<uint>((n * 4 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->rotations().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
      (float4*) thrust::raw_pointer_cast(m_rotations_first.data()),
      (float4*) thrust::raw_pointer_cast(m_rotations_second.data()),
      m_adam_params,
      m_params.rotations_lr * m_global_lr,
      n * 4,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // Scales
    adam_f32x4<<<div_round_up<uint>((n * 3 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->scales().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
      (float4*) thrust::raw_pointer_cast(m_scales_first.data()),
      (float4*) thrust::raw_pointer_cast(m_scales_second.data()),
      m_adam_params,
      m_params.scales_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      ScaleDecay(m_params.scales_l1 * g_scale)
    );

    // SH Coefficient 0
    adam_f32x4<<<div_round_up<uint>((n * 3 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
      m_adam_params,
      m_params.shs_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // SH Coefficients Rest
    const int sh_rest_size = n * (kMaxSphericalHarmonicsCoefficients - 1) * 3;
    adam_f32x4<<<div_round_up<uint>((sh_rest_size + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      sh_rest_size,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );
    maybe_sync(stream);
  }
}


void Adam::step_adamw(float scale, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
  const float gradient_scale = scale;  // This is the gradient scaler, not learning rate multiplier

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("Adam::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("Adam::step: gaussians and gaussians_grad must have same size");
  }

  auto n = m_gaussians->size();
  const int grid = (n + block_size - 1) / block_size;

  float g_scale = 1.0f;
  if (m_adam_params.decay_reduction == "mean") {
    g_scale = 1.0f / n;
  }

  m_global_steps++;
  const float bias_correction1 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                     static_cast<double>(m_global_steps)));
  const float bias_correction2_sqrt = static_cast<float>(
      std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                               static_cast<double>(m_global_steps))));

  const float scene_scale = m_gaussians->scene_scale();
  {
    auto msg = regstr::get<m_step>();
    nvtx3::event_attributes attr(msg, nvtx3::payload{n});
    range range(attr);
    adamw_f32x4<<<div_round_up<uint>((n * 3 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->means().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
      (float4*) thrust::raw_pointer_cast(m_means_first.data()),
      (float4*) thrust::raw_pointer_cast(m_means_second.data()),
      m_adam_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // Opacities
    adamw<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_opacities_first.data()),
      (float*) thrust::raw_pointer_cast(m_opacities_second.data()),
      m_adam_params,
      m_params.opacities_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      OpacityDecay(m_params.opacities_l1 * g_scale)
    );

    // Rotations
    adamw_f32x4<<<div_round_up<uint>((n * 4 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->rotations().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
      (float4*) thrust::raw_pointer_cast(m_rotations_first.data()),
      (float4*) thrust::raw_pointer_cast(m_rotations_second.data()),
      m_adam_params,
      m_params.rotations_lr * m_global_lr,
      n * 4,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // Scales
    adamw_f32x4<<<div_round_up<uint>((n * 3 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->scales().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
      (float4*) thrust::raw_pointer_cast(m_scales_first.data()),
      (float4*) thrust::raw_pointer_cast(m_scales_second.data()),
      m_adam_params,
      m_params.scales_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      ScaleDecay(m_params.scales_l1 * g_scale)
    );

    // SH Coefficient 0
    adamw_f32x4<<<div_round_up<uint>((n * 3 + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
      m_adam_params,
      m_params.shs_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );

    // SH Coefficients Rest
    const int sh_rest_size = n * (kMaxSphericalHarmonicsCoefficients - 1) * 3;
    adamw_f32x4<<<div_round_up<uint>((sh_rest_size + 3) / 4, block_size), block_size, 0, stream>>>(
      (float4*) thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
      (float4*) thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
      (float4*) thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      sh_rest_size,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1
    );
    maybe_sync(stream);
  }
}


Adam::Adam(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
  OptimizerBase(gaussians, gaussians_grad) {
  // Resize and reset all internal buffers
  Adam::reset();
}

__global__ void copy_optimizer_state(
  const vec3 * __restrict__ src_means_first,
  const vec3 * __restrict__ src_means_second,
  vec3 * __restrict__ dst_means_first,
  vec3 * __restrict__ dst_means_second,
  const float * __restrict__ src_opacities_first,
  const float * __restrict__ src_opacities_second,
  float * __restrict__ dst_opacities_first,
  float * __restrict__ dst_opacities_second,
  const vec4 * __restrict__ src_rotations_first,
  const vec4 * __restrict__ src_rotations_second,
  vec4 * __restrict__ dst_rotations_first,
  vec4 * __restrict__ dst_rotations_second,
  const vec3 * __restrict__ src_scales_first,
  const vec3 * __restrict__ src_scales_second,
  vec3 * __restrict__ dst_scales_first,
  vec3 * __restrict__ dst_scales_second,
  const vec3 * __restrict__ src_sh_coefficient_0_first,
  const vec3 * __restrict__ src_sh_coefficient_0_second,
  vec3 * __restrict__ dst_sh_coefficient_0_first,
  vec3 * __restrict__ dst_sh_coefficient_0_second,
  const vec3 * __restrict__ src_sh_coefficients_rest_first,
  const vec3 * __restrict__ src_sh_coefficients_rest_second,
  vec3 * __restrict__ dst_sh_coefficients_rest_first,
  vec3 * __restrict__ dst_sh_coefficients_rest_second,
  const uint * __restrict__ mapping,
  int num_items
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  uint src_idx = mapping[idx];

  // Copy first/second moments for means
  dst_means_first[idx] = src_means_first[src_idx];
  dst_means_second[idx] = src_means_second[src_idx];

  // Copy first/second moments for opacities
  dst_opacities_first[idx] = src_opacities_first[src_idx];
  dst_opacities_second[idx] = src_opacities_second[src_idx];

  // Copy first/second moments for rotations
  dst_rotations_first[idx] = src_rotations_first[src_idx];
  dst_rotations_second[idx] = src_rotations_second[src_idx];

  // Copy first/second moments for scales
  dst_scales_first[idx] = src_scales_first[src_idx];
  dst_scales_second[idx] = src_scales_second[src_idx];

  // Copy first/second moments for SH coefficient 0
  dst_sh_coefficient_0_first[idx] = src_sh_coefficient_0_first[src_idx];
  dst_sh_coefficient_0_second[idx] = src_sh_coefficient_0_second[src_idx];

  // Copy first/second moments for rest SH coefficients
  int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
    dst_sh_coefficients_rest_first[dst_rest_start + i] = src_sh_coefficients_rest_first[src_rest_start + i];
    dst_sh_coefficients_rest_second[dst_rest_start + i] = src_sh_coefficients_rest_second[src_rest_start + i];
  }
}

// Legacy kernel for backward compatibility
__global__ void copy_items(
  const vec3 * __restrict__ src_means_first,
  const vec3 * __restrict__ src_means_second,
  vec3 * __restrict__ dst_means_first,
  vec3 * __restrict__ dst_means_second,
  const float * __restrict__ src_opacities_first,
  const float * __restrict__ src_opacities_second,
  float * __restrict__ dst_opacities_first,
  float * __restrict__ dst_opacities_second,
  const vec4 * __restrict__ src_rotations_first,
  const vec4 * __restrict__ src_rotations_second,
  vec4 * __restrict__ dst_rotations_first,
  vec4 * __restrict__ dst_rotations_second,
  const vec3 * __restrict__ src_scales_first,
  const vec3 * __restrict__ src_scales_second,
  vec3 * __restrict__ dst_scales_first,
  vec3 * __restrict__ dst_scales_second,
  const vec3 * __restrict__ src_sh_coefficient_0_first,
  const vec3 * __restrict__ src_sh_coefficient_0_second,
  vec3 * __restrict__ dst_sh_coefficient_0_first,
  vec3 * __restrict__ dst_sh_coefficient_0_second,
  const vec3 * __restrict__ src_sh_coefficients_rest_first,
  const vec3 * __restrict__ src_sh_coefficients_rest_second,
  vec3 * __restrict__ dst_sh_coefficients_rest_first,
  vec3 * __restrict__ dst_sh_coefficients_rest_second,
  const int * __restrict__ mapping,
  int num_kept
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kept) return;

  int src_idx = mapping[idx];

  // Copy first/second moments for means
  dst_means_first[idx] = src_means_first[src_idx];
  dst_means_second[idx] = src_means_second[src_idx];

  // Copy first/second moments for opacities
  dst_opacities_first[idx] = src_opacities_first[src_idx];
  dst_opacities_second[idx] = src_opacities_second[src_idx];

  // Copy first/second moments for rotations
  dst_rotations_first[idx] = src_rotations_first[src_idx];
  dst_rotations_second[idx] = src_rotations_second[src_idx];

  // Copy first/second moments for scales
  dst_scales_first[idx] = src_scales_first[src_idx];
  dst_scales_second[idx] = src_scales_second[src_idx];

  // Copy first/second moments for SH coefficient 0
  dst_sh_coefficient_0_first[idx] = src_sh_coefficient_0_first[src_idx];
  dst_sh_coefficient_0_second[idx] = src_sh_coefficient_0_second[src_idx];

  // Copy first/second moments for rest SH coefficients
  int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
    dst_sh_coefficients_rest_first[dst_rest_start + i] = src_sh_coefficients_rest_first[src_rest_start + i];
    dst_sh_coefficients_rest_second[dst_rest_start + i] = src_sh_coefficients_rest_second[src_rest_start + i];
  }
}

void Adam::remove(char* kept_flag, int num_kept) {
  // filters the gaussians' first second.
  size_t original_size = m_gaussians->size();
  thrust::device_vector<uint> mapping(original_size); // mapping[idx] = original_idx

  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<uint>(0), thrust::make_counting_iterator<uint>(original_size),
    mapping.begin(), [kept_flag] __device__ (uint orig) { return static_cast<bool>(kept_flag[orig]); });

  thrust::device_vector<vec3> means_first(num_kept);
  thrust::device_vector<vec3> means_second(num_kept);
  thrust::device_vector<float> opacities_first(num_kept);
  thrust::device_vector<float> opacities_second(num_kept);
  thrust::device_vector<vec4> rotations_first(num_kept);
  thrust::device_vector<vec4> rotations_second(num_kept);
  thrust::device_vector<vec3> scales_first(num_kept);
  thrust::device_vector<vec3> scales_second(num_kept);
  thrust::device_vector<vec3> sh_coefficients_0_first(num_kept);
  thrust::device_vector<vec3> sh_coefficients_0_second(num_kept);
  thrust::device_vector<vec3> sh_coefficients_rest_first(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));
  thrust::device_vector<vec3> sh_coefficients_rest_second(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));

  const int grid = (num_kept + block_size - 1) / block_size;
  copy_optimizer_state<<<grid, block_size>>>(
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
      thrust::raw_pointer_cast(sh_coefficients_0_first.data()),
      thrust::raw_pointer_cast(sh_coefficients_0_second.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest_first.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest_second.data()),
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
  m_sh_coefficient_0_first = std::move(sh_coefficients_0_first);
  m_sh_coefficient_0_second = std::move(sh_coefficients_0_second);
  m_sh_coefficients_rest_first = std::move(sh_coefficients_rest_first);
  m_sh_coefficients_rest_second = std::move(sh_coefficients_rest_second);
}

void Adam::duplicate(int* indices, int* new_indices, int num_duplicate) {
  if (num_duplicate == 0) return;
  const uint32_t num_sh_rest_per_gaussian = kMaxSphericalHarmonicsCoefficients - 1;
  // Resize vectors to accommodate duplicated gaussians
  m_means_first.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_means_second.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_opacities_first.resize(m_gaussians->size(), 0.f);
  m_opacities_second.resize(m_gaussians->size(), 0.f);
  m_rotations_first.resize(m_gaussians->size(), vec4(0.f, 0.f, 0.f, 0.f));
  m_rotations_second.resize(m_gaussians->size(), vec4(0.f, 0.f, 0.f, 0.f));
  m_scales_first.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_scales_second.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_sh_coefficient_0_first.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_sh_coefficient_0_second.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_sh_coefficients_rest_first.resize(m_gaussians->size() * num_sh_rest_per_gaussian, vec3(0.f, 0.f, 0.f));
  m_sh_coefficients_rest_second.resize(m_gaussians->size() * num_sh_rest_per_gaussian, vec3(0.f, 0.f, 0.f));
}

void Adam::reset() {
  size_t num_gaussians = m_gaussians->size();
  m_global_steps = 0;

  m_means_first.resize(num_gaussians, vec3(0.f));
  m_means_second.resize(num_gaussians, vec3(0.f));
  m_opacities_first.resize(num_gaussians, 0.f);
  m_opacities_second.resize(num_gaussians, 0.f);
  m_rotations_first.resize(num_gaussians, vec4(0.f));
  m_rotations_second.resize(num_gaussians, vec4(0.f));
  m_scales_first.resize(num_gaussians, vec3(0.f));
  m_scales_second.resize(num_gaussians, vec3(0.f));
  m_sh_coefficient_0_first.resize(num_gaussians, vec3(0.f));
  m_sh_coefficient_0_second.resize(num_gaussians, vec3(0.f));
  m_sh_coefficients_rest_first.resize(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
  m_sh_coefficients_rest_second.resize(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
}

void Adam::reset(int* indices, int num_reset) {
  size_t num_gaussians = m_gaussians->size();
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
      sh_coefficient_0_first = m_sh_coefficient_0_first.data(),
      sh_coefficient_0_second = m_sh_coefficient_0_second.data(),
      sh_coefficients_rest_first = m_sh_coefficients_rest_first.data(),
      sh_coefficients_rest_second = m_sh_coefficients_rest_second.data()
    ] __device__(int idx) {
      means_first[idx] = vec3(0.f);
      means_second[idx] = vec3(0.f);
      opacities_first[idx] = 0.f;
      opacities_second[idx] = 0.f;
      rotations_first[idx] = vec4(0.f);
      rotations_second[idx] = vec4(0.f);
      scales_first[idx] = vec3(0.f);
      scales_second[idx] = vec3(0.f);
      sh_coefficient_0_first[idx] = vec3(0.f);
      sh_coefficient_0_second[idx] = vec3(0.f);
      for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
        sh_coefficients_rest_first[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
        sh_coefficients_rest_second[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
      }
    }
  );
}

void Adam::reset_opacity() {
  // fxxk.
  thrust::fill(m_opacities_first.begin(), m_opacities_first.end(), 0.f);
  thrust::fill(m_opacities_second.begin(), m_opacities_second.end(), 0.f);
}

void Adam::set_params(const json& config) {
  // Update base optimizer parameters
  OptimizerBase::set_params(config);

  // Update Adam-specific parameters
  m_adam_params.from_json(config);
}

json Adam::get_params() const {
  // Get base optimizer parameters
  json params = OptimizerBase::get_params();

  // Add type for reflection
  params["type"] = "adam";

  // Add Adam-specific parameters
  json Adam_params = m_adam_params.to_json();

  // Merge the two JSON objects
  for (auto& [key, value] : Adam_params.items()) {
    params[key] = value;
  }

  return params;
}

json AdamParameters::to_json() const {
  json j;
  j["beta1"] = beta1;
  j["beta2"] = beta2;
  j["epsilon"] = epsilon;
  // Added logging intervals (global, not struct fields)
  j["momentum_log_interval"] = g_momentum_log_interval;
  j["gradient_log_interval"] = g_gradient_log_interval;
  j["decouple_decay"] = decouple_decay;
  j["decay_reduction"] = decay_reduction;
  return j;
}

void AdamParameters::from_json(const json& config) {
  if (config.contains("beta1")) beta1 = config.at("beta1").get<float>();
  if (config.contains("beta2")) beta2 = config.at("beta2").get<float>();
  if (config.contains("epsilon")) epsilon = config.at("epsilon").get<float>();
  if (config.contains("momentum_log_interval")) g_momentum_log_interval = config.at("momentum_log_interval").get<int>();
  if (config.contains("gradient_log_interval")) g_gradient_log_interval = config.at("gradient_log_interval").get<int>();
  if (config.contains("decouple_decay")) decouple_decay = config.at("decouple_decay").get<bool>();
  if (config.contains("decay_reduction")) decay_reduction = config.at("decay_reduction").get<std::string>();
}

/// @brief Reorder Gaussians based on provided indices
void Adam::reorder(uint* indices) {
  int num_gaussians = m_means_first.size();
  
  // Create temporary vectors for reordered data
  thrust::device_vector<vec3> means_first(num_gaussians);
  thrust::device_vector<vec3> means_second(num_gaussians);
  thrust::device_vector<float> opacities_first(num_gaussians);
  thrust::device_vector<float> opacities_second(num_gaussians);
  thrust::device_vector<vec4> rotations_first(num_gaussians);
  thrust::device_vector<vec4> rotations_second(num_gaussians);
  thrust::device_vector<vec3> scales_first(num_gaussians);
  thrust::device_vector<vec3> scales_second(num_gaussians);
  thrust::device_vector<vec3> sh_coefficients_0_first(num_gaussians);
  thrust::device_vector<vec3> sh_coefficients_0_second(num_gaussians);
  thrust::device_vector<vec3> sh_coefficients_rest_first(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1));
  thrust::device_vector<vec3> sh_coefficients_rest_second(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1));

  const int grid = (num_gaussians + block_size - 1) / block_size;
  copy_optimizer_state<<<grid, block_size>>>(
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
      thrust::raw_pointer_cast(sh_coefficients_0_first.data()),
      thrust::raw_pointer_cast(sh_coefficients_0_second.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest_first.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest_second.data()),
      indices,
      num_gaussians
  );

  // Move reordered data back to member variables
  m_means_first = std::move(means_first);
  m_means_second = std::move(means_second);
  m_opacities_first = std::move(opacities_first);
  m_opacities_second = std::move(opacities_second);
  m_rotations_first = std::move(rotations_first);
  m_rotations_second = std::move(rotations_second);
  m_scales_first = std::move(scales_first);
  m_scales_second = std::move(scales_second);
  m_sh_coefficient_0_first = std::move(sh_coefficients_0_first);
  m_sh_coefficient_0_second = std::move(sh_coefficients_0_second);
  m_sh_coefficients_rest_first = std::move(sh_coefficients_rest_first);
  m_sh_coefficients_rest_second = std::move(sh_coefficients_rest_second);
}

} // namespace tinygs
