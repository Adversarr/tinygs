#include <cuda/pipeline>
#include <thrust/execution_policy.h>
#include <cuda/barrier>
#include <nvtx3/nvtx3.hpp>
#include <cooperative_groups.h>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adam.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <cstdio>
#include <cstdlib>

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

template <typename DecayFunc = NoDecay>
static inline void launch_adam_mixed(
    float* theta,
    const float* theta_grad,
    float* theta_first,
    float* theta_second,
    const AdamParameters& adam_p,
    float lr,
    uint32_t num_elements,
    float gradient_scale,
    float bias_correction1,
    float bias_correction2_sqrt,
    float max_grad_1,
    cudaStream_t stream,
    DecayFunc decay = DecayFunc()) {
  if (num_elements == 0) return;

  const uint32_t vec4_elems = num_elements / 4;
  const uint32_t vec4_span = vec4_elems * 4;
  if (vec4_elems > 0) {
    adam_f32x4<<<div_round_up<uint>(vec4_elems, block_size), block_size, 0, stream>>>(
        reinterpret_cast<float4*>(theta),
        reinterpret_cast<const float4*>(theta_grad),
        reinterpret_cast<float4*>(theta_first),
        reinterpret_cast<float4*>(theta_second),
        adam_p,
        lr,
        vec4_span,
        gradient_scale,
        bias_correction1,
        bias_correction2_sqrt,
        max_grad_1,
        decay);
  }

  const uint32_t remain = num_elements - vec4_span;
  if (remain > 0) {
    adam<<<div_round_up<uint>(remain, block_size), block_size, 0, stream>>>(
        theta + vec4_span,
        theta_grad + vec4_span,
        theta_first + vec4_span,
        theta_second + vec4_span,
        adam_p,
        lr,
        remain,
        gradient_scale,
        bias_correction1,
        bias_correction2_sqrt,
        max_grad_1,
        decay);
  }
}

template <typename DecayFunc = NoDecay>
static inline void launch_adamw_mixed(
    float* theta,
    const float* theta_grad,
    float* theta_first,
    float* theta_second,
    const AdamParameters& adam_p,
    float lr,
    uint32_t num_elements,
    float gradient_scale,
    float bias_correction1,
    float bias_correction2_sqrt,
    float max_grad_1,
    cudaStream_t stream,
    DecayFunc decay = DecayFunc()) {
  if (num_elements == 0) return;

  const uint32_t vec4_elems = num_elements / 4;
  const uint32_t vec4_span = vec4_elems * 4;
  if (vec4_elems > 0) {
    adamw_f32x4<<<div_round_up<uint>(vec4_elems, block_size), block_size, 0, stream>>>(
        reinterpret_cast<float4*>(theta),
        reinterpret_cast<const float4*>(theta_grad),
        reinterpret_cast<float4*>(theta_first),
        reinterpret_cast<float4*>(theta_second),
        adam_p,
        lr,
        vec4_span,
        gradient_scale,
        bias_correction1,
        bias_correction2_sqrt,
        max_grad_1,
        decay);
  }

  const uint32_t remain = num_elements - vec4_span;
  if (remain > 0) {
    adamw<<<div_round_up<uint>(remain, block_size), block_size, 0, stream>>>(
        theta + vec4_span,
        theta_grad + vec4_span,
        theta_first + vec4_span,
        theta_second + vec4_span,
        adam_p,
        lr,
        remain,
        gradient_scale,
        bias_correction1,
        bias_correction2_sqrt,
        max_grad_1,
        decay);
  }
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

bool optimizer_kernel_debug_enabled() {
  static int enabled = []() {
    const char* v = std::getenv("TINYGS_DEBUG_OPTIMIZER_KERNELS");
    return (v && (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE")) ? 1 : 0;
  }();
  return enabled != 0;
}

inline void debug_optimizer_kernel(cudaStream_t stream, uint64_t step, const char* name) {
  if (!optimizer_kernel_debug_enabled()) return;
  CUDA_CHECK_THROW(cudaPeekAtLastError());
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
  log_info("[Adam-DBG] step={} kernel={}", step, name);
}

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
      m_sh0_l1       = l1_vec(m_sh0_first, FloatAbs{});
      m_shrest_l1    = 0.f; // TODO: aggregate per-degree SH logging
    }

    // Gradient L1
    float g_means_l1 = 0.f, g_opacities_l1 = 0.f, g_rot_l1 = 0.f, g_scales_l1 = 0.f, g_sh0_l1 = 0.f, g_shrest_l1 = 0.f;
    if (m_global_steps % g_gradient_log_interval == 0) {
      g_means_l1     = l1_vec(m_gaussians_grad->means(), Vec3AbsSum{});
      g_opacities_l1 = l1_vec(m_gaussians_grad->opacities(), FloatAbs{});
      g_rot_l1       = l1_vec(m_gaussians_grad->rotations(), Vec4AbsSum{});
      g_scales_l1    = l1_vec(m_gaussians_grad->scales(), Vec3AbsSum{});
      g_sh0_l1       = l1_vec(m_gaussians_grad->sh0(), FloatAbs{});
      g_shrest_l1    = 0.f; // TODO: aggregate per-degree SH logging
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
    launch_adam_mixed(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians_grad->means().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_second.data())),
      m_adam_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);
    debug_optimizer_kernel(stream, m_global_steps, "means");

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
    debug_optimizer_kernel(stream, m_global_steps, "opacities");

    // Rotations
    launch_adam_mixed(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians_grad->rotations().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_second.data())),
      m_adam_params,
      m_params.rotations_lr * m_global_lr,
      n * 4,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);
    debug_optimizer_kernel(stream, m_global_steps, "rotations");

    // Scales
    launch_adam_mixed(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians_grad->scales().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_second.data())),
      m_adam_params,
      m_params.scales_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream,
      ScaleDecay(m_params.scales_l1 * g_scale));
    debug_optimizer_kernel(stream, m_global_steps, "scales");

    // SH degree 0 (1 coefficient, 3*N floats)
    launch_adam_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh0().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh0().data()),
      thrust::raw_pointer_cast(m_sh0_first.data()),
      thrust::raw_pointer_cast(m_sh0_second.data()),
      m_adam_params,
      m_params.shs_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);
    debug_optimizer_kernel(stream, m_global_steps, "sh0");

    // SH degree 1 (3 coefficients, 9*N floats)
    launch_adam_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh1().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh1().data()),
      thrust::raw_pointer_cast(m_sh1_first.data()),
      thrust::raw_pointer_cast(m_sh1_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      n * 9,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);
    debug_optimizer_kernel(stream, m_global_steps, "sh1");

    // SH degree 2 (5 coefficients, 15*N floats)
    launch_adam_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh2().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh2().data()),
      thrust::raw_pointer_cast(m_sh2_first.data()),
      thrust::raw_pointer_cast(m_sh2_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      n * 15,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);
    debug_optimizer_kernel(stream, m_global_steps, "sh2");

    // SH degree 3 (7 coefficients, 21*N floats)
    launch_adam_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh3().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh3().data()),
      thrust::raw_pointer_cast(m_sh3_first.data()),
      thrust::raw_pointer_cast(m_sh3_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      n * 21,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);
    debug_optimizer_kernel(stream, m_global_steps, "sh3");
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
    launch_adamw_mixed(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians_grad->means().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_second.data())),
      m_adam_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);

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
    launch_adamw_mixed(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians_grad->rotations().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_second.data())),
      m_adam_params,
      m_params.rotations_lr * m_global_lr,
      n * 4,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);

    // Scales
    launch_adamw_mixed(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians_grad->scales().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_second.data())),
      m_adam_params,
      m_params.scales_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream,
      ScaleDecay(m_params.scales_l1 * g_scale));

    // SH degree 0 (1 coefficient, 3*N floats)
    launch_adamw_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh0().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh0().data()),
      thrust::raw_pointer_cast(m_sh0_first.data()),
      thrust::raw_pointer_cast(m_sh0_second.data()),
      m_adam_params,
      m_params.shs_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);

    // SH degree 1 (3 coefficients, 9*N floats)
    launch_adamw_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh1().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh1().data()),
      thrust::raw_pointer_cast(m_sh1_first.data()),
      thrust::raw_pointer_cast(m_sh1_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      n * 9,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);

    // SH degree 2 (5 coefficients, 15*N floats)
    launch_adamw_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh2().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh2().data()),
      thrust::raw_pointer_cast(m_sh2_first.data()),
      thrust::raw_pointer_cast(m_sh2_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      n * 15,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);

    // SH degree 3 (7 coefficients, 21*N floats)
    launch_adamw_mixed(
      thrust::raw_pointer_cast(m_gaussians->sh3().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh3().data()),
      thrust::raw_pointer_cast(m_sh3_first.data()),
      thrust::raw_pointer_cast(m_sh3_second.data()),
      m_adam_params,
      m_params.shs_lr * kShRestScale * m_global_lr,
      n * 21,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1,
      stream);
    maybe_sync(stream);
  }
}


Adam::Adam(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
  OptimizerBase(gaussians, gaussians_grad) {
  // Resize and reset all internal buffers
  Adam::reset();
}

// SoA float gather for optimizer momentum buffers.
// Copies: dst[ch * num_items + item_idx] = src[ch * src_stride + mapping[item_idx]]
// for all items and channels.
__global__ static void gather_soa_optim(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const uint* __restrict__ mapping,
    int num_items,
    int num_channels,
    int src_stride
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_items * num_channels) return;
  int item_idx = tid % num_items;
  int channel = tid / num_items;
  dst[channel * num_items + item_idx] = src[channel * src_stride + mapping[item_idx]];
}

// Host helper: gather SoA optimizer momentum buffers (first and second moments) for one SH degree.
static void gather_soa_optim_buffers(
    const thrust::device_vector<float>& src_first,
    const thrust::device_vector<float>& src_second,
    thrust::device_vector<float>& dst_first,
    thrust::device_vector<float>& dst_second,
    const uint* mapping,
    int num_items,
    int num_channels,
    int src_stride
) {
  int total = num_items * num_channels;
  dst_first.resize(total, 0.f);
  dst_second.resize(total, 0.f);
  if (total == 0) return;
  const int grid = (total + block_size - 1) / block_size;
  gather_soa_optim<<<grid, block_size>>>(
      thrust::raw_pointer_cast(src_first.data()),
      thrust::raw_pointer_cast(dst_first.data()),
      mapping, num_items, num_channels, src_stride);
  gather_soa_optim<<<grid, block_size>>>(
      thrust::raw_pointer_cast(src_second.data()),
      thrust::raw_pointer_cast(dst_second.data()),
      mapping, num_items, num_channels, src_stride);
}

// Host helper: re-layout a SoA optimizer momentum buffer when the Gaussian count changes.
// Preserves the first old_n items per channel, zero-fills the rest.
static void relayout_soa_optim(
    thrust::device_vector<float>& buf,
    int old_n,
    int new_n,
    int num_channels
) {
  if (old_n == 0 || new_n == 0) {
    buf.assign(new_n * num_channels, 0.f);
    return;
  }
  thrust::device_vector<uint> identity(old_n);
  thrust::sequence(identity.begin(), identity.end());
  thrust::device_vector<float> new_buf(new_n * num_channels, 0.f);
  int total = old_n * num_channels;
  const int grid = (total + block_size - 1) / block_size;
  gather_soa_optim<<<grid, block_size>>>(
      thrust::raw_pointer_cast(buf.data()),
      thrust::raw_pointer_cast(new_buf.data()),
      thrust::raw_pointer_cast(identity.data()),
      old_n, num_channels, old_n);
  buf = std::move(new_buf);
}

// Gather non-SH optimizer state (means, opacities, rotations, scales) using a mapping.
__global__ void copy_optimizer_base_state(
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
}

void Adam::remove(char* kept_flag, int num_kept) {
  // filters the gaussians' first second.
  size_t original_size = m_means_first.size();
  thrust::device_vector<uint> mapping(original_size); // mapping[idx] = original_idx

  auto end_it = thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<uint>(0), thrust::make_counting_iterator<uint>(original_size),
    mapping.begin(), [kept_flag] __device__ (uint orig) { return static_cast<bool>(kept_flag[orig]); });

  // Gather non-SH optimizer state
  thrust::device_vector<vec3> means_first(num_kept);
  thrust::device_vector<vec3> means_second(num_kept);
  thrust::device_vector<float> opacities_first(num_kept);
  thrust::device_vector<float> opacities_second(num_kept);
  thrust::device_vector<vec4> rotations_first(num_kept);
  thrust::device_vector<vec4> rotations_second(num_kept);
  thrust::device_vector<vec3> scales_first(num_kept);
  thrust::device_vector<vec3> scales_second(num_kept);

  const int grid = (num_kept + block_size - 1) / block_size;
  copy_optimizer_base_state<<<grid, block_size>>>(
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

  // Gather per-degree SoA SH optimizer momentum buffers
  const uint* map_ptr = thrust::raw_pointer_cast(mapping.data());
  const int old_n = static_cast<int>(original_size);
  thrust::device_vector<float> sh0_f, sh0_s, sh1_f, sh1_s, sh2_f, sh2_s, sh3_f, sh3_s;
  gather_soa_optim_buffers(m_sh0_first, m_sh0_second, sh0_f, sh0_s, map_ptr, num_kept,
                           GPUGaussian3d::sh_degree_num_coeffs(0) * 3, old_n);
  gather_soa_optim_buffers(m_sh1_first, m_sh1_second, sh1_f, sh1_s, map_ptr, num_kept,
                           GPUGaussian3d::sh_degree_num_coeffs(1) * 3, old_n);
  gather_soa_optim_buffers(m_sh2_first, m_sh2_second, sh2_f, sh2_s, map_ptr, num_kept,
                           GPUGaussian3d::sh_degree_num_coeffs(2) * 3, old_n);
  gather_soa_optim_buffers(m_sh3_first, m_sh3_second, sh3_f, sh3_s, map_ptr, num_kept,
                           GPUGaussian3d::sh_degree_num_coeffs(3) * 3, old_n);
  m_sh0_first = std::move(sh0_f); m_sh0_second = std::move(sh0_s);
  m_sh1_first = std::move(sh1_f); m_sh1_second = std::move(sh1_s);
  m_sh2_first = std::move(sh2_f); m_sh2_second = std::move(sh2_s);
  m_sh3_first = std::move(sh3_f); m_sh3_second = std::move(sh3_s);
}

void Adam::duplicate(int* indices, int* new_indices, int num_duplicate) {
  if (num_duplicate == 0) return;

  // Save old Gaussian count before resize (SoA stride changes)
  const int old_n = static_cast<int>(m_means_first.size());
  const int new_n = static_cast<int>(m_gaussians->size());

  // Non-SH fields: simple resize with zero-fill (contiguous per-Gaussian, no stride change)
  m_means_first.resize(new_n, vec3(0.f, 0.f, 0.f));
  m_means_second.resize(new_n, vec3(0.f, 0.f, 0.f));
  m_opacities_first.resize(new_n, 0.f);
  m_opacities_second.resize(new_n, 0.f);
  m_rotations_first.resize(new_n, vec4(0.f, 0.f, 0.f, 0.f));
  m_rotations_second.resize(new_n, vec4(0.f, 0.f, 0.f, 0.f));
  m_scales_first.resize(new_n, vec3(0.f, 0.f, 0.f));
  m_scales_second.resize(new_n, vec3(0.f, 0.f, 0.f));

  // SH fields: SoA layout means the stride (N) changes, so we must re-layout.
  // New Gaussians get zero momentum.
  for (int deg = 0; deg < 4; deg++) {
    int nc = GPUGaussian3d::sh_degree_num_coeffs(deg) * 3; // num SoA channels
    auto relayout_pair = [&](thrust::device_vector<float>& first, thrust::device_vector<float>& second) {
      relayout_soa_optim(first, old_n, new_n, nc);
      relayout_soa_optim(second, old_n, new_n, nc);
    };
    switch (deg) {
      case 0: relayout_pair(m_sh0_first, m_sh0_second); break;
      case 1: relayout_pair(m_sh1_first, m_sh1_second); break;
      case 2: relayout_pair(m_sh2_first, m_sh2_second); break;
      case 3: relayout_pair(m_sh3_first, m_sh3_second); break;
    }
  }
}

void Adam::reset() {
  size_t num_gaussians = m_gaussians->size();
  m_global_steps = 0;

  m_means_first.assign(num_gaussians, vec3(0.f));
  m_means_second.assign(num_gaussians, vec3(0.f));
  m_opacities_first.assign(num_gaussians, 0.f);
  m_opacities_second.assign(num_gaussians, 0.f);
  m_rotations_first.assign(num_gaussians, vec4(0.f));
  m_rotations_second.assign(num_gaussians, vec4(0.f));
  m_scales_first.assign(num_gaussians, vec3(0.f));
  m_scales_second.assign(num_gaussians, vec3(0.f));
  m_sh0_first.assign(num_gaussians * 3, 0.f);
  m_sh0_second.assign(num_gaussians * 3, 0.f);
  m_sh1_first.assign(num_gaussians * 9, 0.f);
  m_sh1_second.assign(num_gaussians * 9, 0.f);
  m_sh2_first.assign(num_gaussians * 15, 0.f);
  m_sh2_second.assign(num_gaussians * 15, 0.f);
  m_sh3_first.assign(num_gaussians * 21, 0.f);
  m_sh3_second.assign(num_gaussians * 21, 0.f);
}

void Adam::reset(int* indices, int num_reset) {
  const int n = static_cast<int>(m_gaussians->size());
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
      sh0_first = thrust::raw_pointer_cast(m_sh0_first.data()),
      sh0_second = thrust::raw_pointer_cast(m_sh0_second.data()),
      sh1_first = thrust::raw_pointer_cast(m_sh1_first.data()),
      sh1_second = thrust::raw_pointer_cast(m_sh1_second.data()),
      sh2_first = thrust::raw_pointer_cast(m_sh2_first.data()),
      sh2_second = thrust::raw_pointer_cast(m_sh2_second.data()),
      sh3_first = thrust::raw_pointer_cast(m_sh3_first.data()),
      sh3_second = thrust::raw_pointer_cast(m_sh3_second.data()),
      n
    ] __device__(int idx) {
      means_first[idx] = vec3(0.f);
      means_second[idx] = vec3(0.f);
      opacities_first[idx] = 0.f;
      opacities_second[idx] = 0.f;
      rotations_first[idx] = vec4(0.f);
      rotations_second[idx] = vec4(0.f);
      scales_first[idx] = vec3(0.f);
      scales_second[idx] = vec3(0.f);
      // Zero per-degree SoA SH moments: data[ch * N + idx] = 0 for all channels
      for (int ch = 0; ch < 1 * 3; ch++) { sh0_first[ch * n + idx] = 0.f; sh0_second[ch * n + idx] = 0.f; }
      for (int ch = 0; ch < 3 * 3; ch++) { sh1_first[ch * n + idx] = 0.f; sh1_second[ch * n + idx] = 0.f; }
      for (int ch = 0; ch < 5 * 3; ch++) { sh2_first[ch * n + idx] = 0.f; sh2_second[ch * n + idx] = 0.f; }
      for (int ch = 0; ch < 7 * 3; ch++) { sh3_first[ch * n + idx] = 0.f; sh3_second[ch * n + idx] = 0.f; }
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

  // Gather non-SH optimizer state
  thrust::device_vector<vec3> means_first(num_gaussians);
  thrust::device_vector<vec3> means_second(num_gaussians);
  thrust::device_vector<float> opacities_first(num_gaussians);
  thrust::device_vector<float> opacities_second(num_gaussians);
  thrust::device_vector<vec4> rotations_first(num_gaussians);
  thrust::device_vector<vec4> rotations_second(num_gaussians);
  thrust::device_vector<vec3> scales_first(num_gaussians);
  thrust::device_vector<vec3> scales_second(num_gaussians);

  const int grid = (num_gaussians + block_size - 1) / block_size;
  copy_optimizer_base_state<<<grid, block_size>>>(
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
      indices,
      num_gaussians
  );

  // Move reordered non-SH data back to member variables
  m_means_first = std::move(means_first);
  m_means_second = std::move(means_second);
  m_opacities_first = std::move(opacities_first);
  m_opacities_second = std::move(opacities_second);
  m_rotations_first = std::move(rotations_first);
  m_rotations_second = std::move(rotations_second);
  m_scales_first = std::move(scales_first);
  m_scales_second = std::move(scales_second);

  // Gather per-degree SoA SH optimizer momentum buffers (stride == num_gaussians for reorder)
  thrust::device_vector<float> sh0_f, sh0_s, sh1_f, sh1_s, sh2_f, sh2_s, sh3_f, sh3_s;
  gather_soa_optim_buffers(m_sh0_first, m_sh0_second, sh0_f, sh0_s, indices, num_gaussians,
                           GPUGaussian3d::sh_degree_num_coeffs(0) * 3, num_gaussians);
  gather_soa_optim_buffers(m_sh1_first, m_sh1_second, sh1_f, sh1_s, indices, num_gaussians,
                           GPUGaussian3d::sh_degree_num_coeffs(1) * 3, num_gaussians);
  gather_soa_optim_buffers(m_sh2_first, m_sh2_second, sh2_f, sh2_s, indices, num_gaussians,
                           GPUGaussian3d::sh_degree_num_coeffs(2) * 3, num_gaussians);
  gather_soa_optim_buffers(m_sh3_first, m_sh3_second, sh3_f, sh3_s, indices, num_gaussians,
                           GPUGaussian3d::sh_degree_num_coeffs(3) * 3, num_gaussians);
  m_sh0_first = std::move(sh0_f); m_sh0_second = std::move(sh0_s);
  m_sh1_first = std::move(sh1_f); m_sh1_second = std::move(sh1_s);
  m_sh2_first = std::move(sh2_f); m_sh2_second = std::move(sh2_s);
  m_sh3_first = std::move(sh3_f); m_sh3_second = std::move(sh3_s);
}

} // namespace tinygs
