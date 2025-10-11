#include <thrust/execution_policy.h>
#include <nvtx3/nvtx3.hpp>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adan.hpp"

namespace tinygs {

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

template<typename DecayFunc = NoDecay>
__global__ void adan_kernel(
    float* __restrict__ thetas,
    const float* __restrict__ thetas_grad,
    float* __restrict__ m_buf,
    float* __restrict__ v_buf,
    float* __restrict__ d_buf,
    float* __restrict__ prev_g_buf,
    AdanParameters ap,
    float lr,
    uint32_t num_elements,
    float gradient_scale,
    float bias_correction1,
    float bias_correction2,
    float bias_correction3_sqrt,
    float max_grad_1,
    float epsilon,
    DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;

  float theta = thetas[idx];
  theta -= lr * f(theta); // decoupled L1 regularization

  float g = gradient_scale * thetas_grad[idx];
  if (max_grad_1 != 0.0f) {
    g = copysignf(fminf(fabsf(g), max_grad_1), g);
  }

  float prev_g = prev_g_buf[idx];
  float diff = g - prev_g;
  float m = m_buf[idx];
  float d = d_buf[idx];
  float v = v_buf[idx];

  m = lerp(m, g, 1.0f - ap.beta1);
  d = lerp(d, diff, 1.0f - ap.beta2);

  const float temp = ap.beta2 * diff + g;
  v = fmaf(1.0f - ap.beta3, temp * temp, ap.beta3 * v);

  const float denom = sqrtf(v) / bias_correction3_sqrt + epsilon;
  const float step_m = lr / bias_correction1;
  const float step_d = lr * ap.beta2 / bias_correction2;

  theta -= step_m * (m / denom);
  theta -= step_d * (d / denom);

  // write back
  thetas[idx] = theta;
  m_buf[idx] = m;
  d_buf[idx] = d;
  v_buf[idx] = v;
  prev_g_buf[idx] = g;
}

template<typename DecayFunc = NoDecay>
__global__ void adan_kernel_vec3(
    float* __restrict__ thetas,
    const float* __restrict__ thetas_grad,
    float* __restrict__ m_buf,
    float* __restrict__ v_buf,
    float* __restrict__ d_buf,
    float* __restrict__ prev_g_buf,
    AdanParameters ap,
    float lr,
    uint32_t num_vec3,
    float gradient_scale,
    float bias_correction1,
    float bias_correction2,
    float bias_correction3_sqrt,
    float max_grad_1,
    float epsilon,
    DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_vec3) return;
  const uint32_t base = idx * 3u;

  vec3 theta = vec3(thetas[base + 0], thetas[base + 1], thetas[base + 2]);
  const vec3 reg = f(theta);
  theta.x -= lr * reg.x;
  theta.y -= lr * reg.y;
  theta.z -= lr * reg.z;

  vec3 g = vec3(gradient_scale * thetas_grad[base + 0],
                 gradient_scale * thetas_grad[base + 1],
                 gradient_scale * thetas_grad[base + 2]);
  if (max_grad_1 != 0.0f) {
    g.x = copysignf(fminf(fabsf(g.x), max_grad_1), g.x);
    g.y = copysignf(fminf(fabsf(g.y), max_grad_1), g.y);
    g.z = copysignf(fminf(fabsf(g.z), max_grad_1), g.z);
  }

  vec3 prev_g = vec3(prev_g_buf[base + 0], prev_g_buf[base + 1], prev_g_buf[base + 2]);
  vec3 diff = vec3(g.x - prev_g.x, g.y - prev_g.y, g.z - prev_g.z);
  vec3 m = vec3(m_buf[base + 0], m_buf[base + 1], m_buf[base + 2]);
  vec3 d = vec3(d_buf[base + 0], d_buf[base + 1], d_buf[base + 2]);
  vec3 v = vec3(v_buf[base + 0], v_buf[base + 1], v_buf[base + 2]);

  m.x = lerp(m.x, g.x, 1.0f - ap.beta1);
  m.y = lerp(m.y, g.y, 1.0f - ap.beta1);
  m.z = lerp(m.z, g.z, 1.0f - ap.beta1);

  d.x = lerp(d.x, diff.x, 1.0f - ap.beta2);
  d.y = lerp(d.y, diff.y, 1.0f - ap.beta2);
  d.z = lerp(d.z, diff.z, 1.0f - ap.beta2);

  const vec3 temp = vec3(ap.beta2 * diff.x + g.x,
                         ap.beta2 * diff.y + g.y,
                         ap.beta2 * diff.z + g.z);
  v.x = fmaf(1.0f - ap.beta3, temp.x * temp.x, ap.beta3 * v.x);
  v.y = fmaf(1.0f - ap.beta3, temp.y * temp.y, ap.beta3 * v.y);
  v.z = fmaf(1.0f - ap.beta3, temp.z * temp.z, ap.beta3 * v.z);

  const float denom_x = sqrtf(v.x) / bias_correction3_sqrt + epsilon;
  const float denom_y = sqrtf(v.y) / bias_correction3_sqrt + epsilon;
  const float denom_z = sqrtf(v.z) / bias_correction3_sqrt + epsilon;
  const float step_m = lr / bias_correction1;
  const float step_d = lr * ap.beta2 / bias_correction2;

  theta.x -= step_m * (m.x / denom_x) + step_d * (d.x / denom_x);
  theta.y -= step_m * (m.y / denom_y) + step_d * (d.y / denom_y);
  theta.z -= step_m * (m.z / denom_z) + step_d * (d.z / denom_z);

  thetas[base + 0] = theta.x; thetas[base + 1] = theta.y; thetas[base + 2] = theta.z;
  m_buf[base + 0] = m.x; m_buf[base + 1] = m.y; m_buf[base + 2] = m.z;
  d_buf[base + 0] = d.x; d_buf[base + 1] = d.y; d_buf[base + 2] = d.z;
  v_buf[base + 0] = v.x; v_buf[base + 1] = v.y; v_buf[base + 2] = v.z;
  prev_g_buf[base + 0] = g.x; prev_g_buf[base + 1] = g.y; prev_g_buf[base + 2] = g.z;
}

template<typename DecayFunc = NoDecay>
__global__ void adan_kernel_vec4(
    float* __restrict__ thetas,
    const float* __restrict__ thetas_grad,
    float* __restrict__ m_buf,
    float* __restrict__ v_buf,
    float* __restrict__ d_buf,
    float* __restrict__ prev_g_buf,
    AdanParameters ap,
    float lr,
    uint32_t num_vec4,
    float gradient_scale,
    float bias_correction1,
    float bias_correction2,
    float bias_correction3_sqrt,
    float max_grad_1,
    float epsilon,
    DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_vec4) return;
  const uint32_t base = idx * 4u;

  vec4 theta = vec4(thetas[base + 0], thetas[base + 1], thetas[base + 2], thetas[base + 3]);
  const vec4 reg = f(theta);
  theta.x -= lr * reg.x; theta.y -= lr * reg.y; theta.z -= lr * reg.z; theta.w -= lr * reg.w;

  vec4 g = vec4(gradient_scale * thetas_grad[base + 0],
                 gradient_scale * thetas_grad[base + 1],
                 gradient_scale * thetas_grad[base + 2],
                 gradient_scale * thetas_grad[base + 3]);
  if (max_grad_1 != 0.0f) {
    g.x = copysignf(fminf(fabsf(g.x), max_grad_1), g.x);
    g.y = copysignf(fminf(fabsf(g.y), max_grad_1), g.y);
    g.z = copysignf(fminf(fabsf(g.z), max_grad_1), g.z);
    g.w = copysignf(fminf(fabsf(g.w), max_grad_1), g.w);
  }

  vec4 prev_g = vec4(prev_g_buf[base + 0], prev_g_buf[base + 1], prev_g_buf[base + 2], prev_g_buf[base + 3]);
  vec4 diff = vec4(g.x - prev_g.x, g.y - prev_g.y, g.z - prev_g.z, g.w - prev_g.w);
  vec4 m = vec4(m_buf[base + 0], m_buf[base + 1], m_buf[base + 2], m_buf[base + 3]);
  vec4 d = vec4(d_buf[base + 0], d_buf[base + 1], d_buf[base + 2], d_buf[base + 3]);
  vec4 v = vec4(v_buf[base + 0], v_buf[base + 1], v_buf[base + 2], v_buf[base + 3]);

  m.x = lerp(m.x, g.x, 1.0f - ap.beta1); m.y = lerp(m.y, g.y, 1.0f - ap.beta1);
  m.z = lerp(m.z, g.z, 1.0f - ap.beta1); m.w = lerp(m.w, g.w, 1.0f - ap.beta1);

  d.x = lerp(d.x, diff.x, 1.0f - ap.beta2); d.y = lerp(d.y, diff.y, 1.0f - ap.beta2);
  d.z = lerp(d.z, diff.z, 1.0f - ap.beta2); d.w = lerp(d.w, diff.w, 1.0f - ap.beta2);

  const vec4 temp = vec4(ap.beta2 * diff.x + g.x,
                         ap.beta2 * diff.y + g.y,
                         ap.beta2 * diff.z + g.z,
                         ap.beta2 * diff.w + g.w);
  v.x = fmaf(1.0f - ap.beta3, temp.x * temp.x, ap.beta3 * v.x);
  v.y = fmaf(1.0f - ap.beta3, temp.y * temp.y, ap.beta3 * v.y);
  v.z = fmaf(1.0f - ap.beta3, temp.z * temp.z, ap.beta3 * v.z);
  v.w = fmaf(1.0f - ap.beta3, temp.w * temp.w, ap.beta3 * v.w);

  const float denom_x = sqrtf(v.x) / bias_correction3_sqrt + epsilon;
  const float denom_y = sqrtf(v.y) / bias_correction3_sqrt + epsilon;
  const float denom_z = sqrtf(v.z) / bias_correction3_sqrt + epsilon;
  const float denom_w = sqrtf(v.w) / bias_correction3_sqrt + epsilon;
  const float step_m = lr / bias_correction1;
  const float step_d = lr * ap.beta2 / bias_correction2;

  theta.x -= step_m * (m.x / denom_x) + step_d * (d.x / denom_x);
  theta.y -= step_m * (m.y / denom_y) + step_d * (d.y / denom_y);
  theta.z -= step_m * (m.z / denom_z) + step_d * (d.z / denom_z);
  theta.w -= step_m * (m.w / denom_w) + step_d * (d.w / denom_w);

  thetas[base + 0] = theta.x; thetas[base + 1] = theta.y; thetas[base + 2] = theta.z; thetas[base + 3] = theta.w;
  m_buf[base + 0] = m.x; m_buf[base + 1] = m.y; m_buf[base + 2] = m.z; m_buf[base + 3] = m.w;
  d_buf[base + 0] = d.x; d_buf[base + 1] = d.y; d_buf[base + 2] = d.z; d_buf[base + 3] = d.w;
  v_buf[base + 0] = v.x; v_buf[base + 1] = v.y; v_buf[base + 2] = v.z; v_buf[base + 3] = v.w;
  prev_g_buf[base + 0] = g.x; prev_g_buf[base + 1] = g.y; prev_g_buf[base + 2] = g.z; prev_g_buf[base + 3] = g.w;
}

struct adan_domain { static constexpr char const* name{"optim"}; };
using range = nvtx3::scoped_range_in<adan_domain>;
using regstr = nvtx3::registered_string_in<adan_domain>;
struct m_step { static constexpr char const* message{"adan_step"}; };

// Copy optimizer state for kept gaussians (used by remove)
__global__ static void copy_adan_state(
  const vec3* __restrict__ src_means_m,
  const vec3* __restrict__ src_means_v,
  const vec3* __restrict__ src_means_d,
  const vec3* __restrict__ src_means_pg,
  vec3* __restrict__ dst_means_m,
  vec3* __restrict__ dst_means_v,
  vec3* __restrict__ dst_means_d,
  vec3* __restrict__ dst_means_pg,
  const float* __restrict__ src_op_m,
  const float* __restrict__ src_op_v,
  const float* __restrict__ src_op_d,
  const float* __restrict__ src_op_pg,
  float* __restrict__ dst_op_m,
  float* __restrict__ dst_op_v,
  float* __restrict__ dst_op_d,
  float* __restrict__ dst_op_pg,
  const vec4* __restrict__ src_rot_m,
  const vec4* __restrict__ src_rot_v,
  const vec4* __restrict__ src_rot_d,
  const vec4* __restrict__ src_rot_pg,
  vec4* __restrict__ dst_rot_m,
  vec4* __restrict__ dst_rot_v,
  vec4* __restrict__ dst_rot_d,
  vec4* __restrict__ dst_rot_pg,
  const vec3* __restrict__ src_sc_m,
  const vec3* __restrict__ src_sc_v,
  const vec3* __restrict__ src_sc_d,
  const vec3* __restrict__ src_sc_pg,
  vec3* __restrict__ dst_sc_m,
  vec3* __restrict__ dst_sc_v,
  vec3* __restrict__ dst_sc_d,
  vec3* __restrict__ dst_sc_pg,
  const vec3* __restrict__ src_sh0_m,
  const vec3* __restrict__ src_sh0_v,
  const vec3* __restrict__ src_sh0_d,
  const vec3* __restrict__ src_sh0_pg,
  vec3* __restrict__ dst_sh0_m,
  vec3* __restrict__ dst_sh0_v,
  vec3* __restrict__ dst_sh0_d,
  vec3* __restrict__ dst_sh0_pg,
  const vec3* __restrict__ src_shrest_m,
  const vec3* __restrict__ src_shrest_v,
  const vec3* __restrict__ src_shrest_d,
  const vec3* __restrict__ src_shrest_pg,
  vec3* __restrict__ dst_shrest_m,
  vec3* __restrict__ dst_shrest_v,
  vec3* __restrict__ dst_shrest_d,
  vec3* __restrict__ dst_shrest_pg,
  const uint* __restrict__ mapping,
  int num_items,
  uint32_t num_sh_rest
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;
  uint src = mapping[idx];
  dst_means_m[idx] = src_means_m[src];
  dst_means_v[idx] = src_means_v[src];
  dst_means_d[idx] = src_means_d[src];
  dst_means_pg[idx] = src_means_pg[src];
  dst_op_m[idx] = src_op_m[src];
  dst_op_v[idx] = src_op_v[src];
  dst_op_d[idx] = src_op_d[src];
  dst_op_pg[idx] = src_op_pg[src];
  dst_rot_m[idx] = src_rot_m[src];
  dst_rot_v[idx] = src_rot_v[src];
  dst_rot_d[idx] = src_rot_d[src];
  dst_rot_pg[idx] = src_rot_pg[src];
  dst_sc_m[idx] = src_sc_m[src];
  dst_sc_v[idx] = src_sc_v[src];
  dst_sc_d[idx] = src_sc_d[src];
  dst_sc_pg[idx] = src_sc_pg[src];
  dst_sh0_m[idx] = src_sh0_m[src];
  dst_sh0_v[idx] = src_sh0_v[src];
  dst_sh0_d[idx] = src_sh0_d[src];
  dst_sh0_pg[idx] = src_sh0_pg[src];
  const int s_base = src * num_sh_rest;
  const int d_base = idx * num_sh_rest;
  for (uint32_t i = 0; i < num_sh_rest; i++) {
    dst_shrest_m[d_base + i] = src_shrest_m[s_base + i];
    dst_shrest_v[d_base + i] = src_shrest_v[s_base + i];
    dst_shrest_d[d_base + i] = src_shrest_d[s_base + i];
    dst_shrest_pg[d_base + i] = src_shrest_pg[s_base + i];
  }
}

void Adan::step(float scale, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
  const float gradient_scale = scale;
  constexpr int block_size = 256;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("Adan::step: gaussians or gaussians_grad is null");
  }
  if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("Adan::step: gaussians and gradients size mismatch");
  }

  auto n = m_gaussians->size();
  m_global_steps += 1;

  const float beta1 = m_adan_params.beta1;
  const float beta2 = m_adan_params.beta2;
  const float beta3 = m_adan_params.beta3;
  const float bias_correction1 = 1.0f - powf(beta1, (float)m_global_steps);
  const float bias_correction2 = 1.0f - powf(beta2, (float)m_global_steps);
  const float bias_correction3_sqrt = sqrtf(1.0f - powf(beta3, (float)m_global_steps));
  const float epsilon = m_adan_params.epsilon;

  const float scene_scale = m_gaussians->scene_scale();

  {
    auto msg = regstr::get<m_step>();
    nvtx3::event_attributes attr(msg, nvtx3::payload{n});
    range range(attr);

    // Means (vec3)
    adan_kernel_vec3<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->means().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
      (float*)thrust::raw_pointer_cast(m_means_m.data()),
      (float*)thrust::raw_pointer_cast(m_means_v.data()),
      (float*)thrust::raw_pointer_cast(m_means_d.data()),
      (float*)thrust::raw_pointer_cast(m_means_prev_g.data()),
      m_adan_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2,
      bias_correction3_sqrt,
      m_params.max_grad_1,
      epsilon
    );

    // Opacities (float) with L1 decay
    adan_kernel<OpacityDecay><<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      (float*)thrust::raw_pointer_cast(m_opacities_m.data()),
      (float*)thrust::raw_pointer_cast(m_opacities_v.data()),
      (float*)thrust::raw_pointer_cast(m_opacities_d.data()),
      (float*)thrust::raw_pointer_cast(m_opacities_prev_g.data()),
      m_adan_params,
      m_params.opacities_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2,
      bias_correction3_sqrt,
      m_params.max_grad_1,
      epsilon,
      OpacityDecay(m_params.opacities_l1)
    );

    // Rotations (vec4)
    adan_kernel_vec4<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->rotations().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
      (float*)thrust::raw_pointer_cast(m_rotations_m.data()),
      (float*)thrust::raw_pointer_cast(m_rotations_v.data()),
      (float*)thrust::raw_pointer_cast(m_rotations_d.data()),
      (float*)thrust::raw_pointer_cast(m_rotations_prev_g.data()),
      m_adan_params,
      m_params.rotations_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2,
      bias_correction3_sqrt,
      m_params.max_grad_1,
      epsilon
    );

    // Scales (vec3) with L1 decay
    adan_kernel_vec3<ScaleDecay><<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->scales().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
      (float*)thrust::raw_pointer_cast(m_scales_m.data()),
      (float*)thrust::raw_pointer_cast(m_scales_v.data()),
      (float*)thrust::raw_pointer_cast(m_scales_d.data()),
      (float*)thrust::raw_pointer_cast(m_scales_prev_g.data()),
      m_adan_params,
      m_params.scales_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2,
      bias_correction3_sqrt,
      m_params.max_grad_1,
      epsilon,
      ScaleDecay(m_params.scales_l1)
    );

    // SH coefficient 0 (vec3)
    adan_kernel_vec3<<<div_round_up<uint>(n, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
      (float*)thrust::raw_pointer_cast(m_sh0_m.data()),
      (float*)thrust::raw_pointer_cast(m_sh0_v.data()),
      (float*)thrust::raw_pointer_cast(m_sh0_d.data()),
      (float*)thrust::raw_pointer_cast(m_sh0_prev_g.data()),
      m_adan_params,
      m_params.shs_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2,
      bias_correction3_sqrt,
      m_params.max_grad_1,
      epsilon
    );

    // SH coefficients rest (vec3 per coeff)
    const int sh_rest_groups = n * (kMaxSphericalHarmonicsCoefficients - 1);
    adan_kernel_vec3<<<div_round_up<uint>(sh_rest_groups, block_size), block_size, 0, stream>>>(
      (float*)thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
      (float*)thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
      (float*)thrust::raw_pointer_cast(m_shrest_m.data()),
      (float*)thrust::raw_pointer_cast(m_shrest_v.data()),
      (float*)thrust::raw_pointer_cast(m_shrest_d.data()),
      (float*)thrust::raw_pointer_cast(m_shrest_prev_g.data()),
      m_adan_params,
      m_params.shs_lr * 0.05f * m_global_lr,
      sh_rest_groups,
      gradient_scale,
      bias_correction1,
      bias_correction2,
      bias_correction3_sqrt,
      m_params.max_grad_1,
      epsilon
    );

    maybe_sync(stream);
  }
}

Adan::Adan(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad)
  : OptimizerBase(gaussians, gaussians_grad) {
  Adan::reset();
}

void Adan::reset() {
  size_t n = m_gaussians->size();
  const uint32_t num_sh_rest = kMaxSphericalHarmonicsCoefficients - 1;
  m_means_m.resize(n, vec3(0.f));
  m_means_v.resize(n, vec3(0.f));
  m_means_d.resize(n, vec3(0.f));
  m_means_prev_g.resize(n, vec3(0.f));

  m_opacities_m.resize(n, 0.f);
  m_opacities_v.resize(n, 0.f);
  m_opacities_d.resize(n, 0.f);
  m_opacities_prev_g.resize(n, 0.f);

  m_rotations_m.resize(n, vec4(0.f));
  m_rotations_v.resize(n, vec4(0.f));
  m_rotations_d.resize(n, vec4(0.f));
  m_rotations_prev_g.resize(n, vec4(0.f));

  m_scales_m.resize(n, vec3(0.f));
  m_scales_v.resize(n, vec3(0.f));
  m_scales_d.resize(n, vec3(0.f));
  m_scales_prev_g.resize(n, vec3(0.f));

  m_sh0_m.resize(n, vec3(0.f));
  m_sh0_v.resize(n, vec3(0.f));
  m_sh0_d.resize(n, vec3(0.f));
  m_sh0_prev_g.resize(n, vec3(0.f));

  m_shrest_m.resize(n * num_sh_rest, vec3(0.f));
  m_shrest_v.resize(n * num_sh_rest, vec3(0.f));
  m_shrest_d.resize(n * num_sh_rest, vec3(0.f));
  m_shrest_prev_g.resize(n * num_sh_rest, vec3(0.f));
}

void Adan::reset(int* indices, int num_reset) {
  thrust::for_each(
    thrust::device_ptr<int>(indices),
    thrust::device_ptr<int>(indices) + num_reset,
    [
      means_m = m_means_m.data(), means_v = m_means_v.data(), means_d = m_means_d.data(), means_pg = m_means_prev_g.data(),
      op_m = m_opacities_m.data(), op_v = m_opacities_v.data(), op_d = m_opacities_d.data(), op_pg = m_opacities_prev_g.data(),
      rot_m = m_rotations_m.data(), rot_v = m_rotations_v.data(), rot_d = m_rotations_d.data(), rot_pg = m_rotations_prev_g.data(),
      sc_m = m_scales_m.data(), sc_v = m_scales_v.data(), sc_d = m_scales_d.data(), sc_pg = m_scales_prev_g.data(),
      sh0_m = m_sh0_m.data(), sh0_v = m_sh0_v.data(), sh0_d = m_sh0_d.data(), sh0_pg = m_sh0_prev_g.data(),
      shrest_m = m_shrest_m.data(), shrest_v = m_shrest_v.data(), shrest_d = m_shrest_d.data(), shrest_pg = m_shrest_prev_g.data()
    ] __device__ (int idx) {
      means_m[idx] = vec3(0.f); means_v[idx] = vec3(0.f); means_d[idx] = vec3(0.f); means_pg[idx] = vec3(0.f);
      op_m[idx] = 0.f; op_v[idx] = 0.f; op_d[idx] = 0.f; op_pg[idx] = 0.f;
      rot_m[idx] = vec4(0.f); rot_v[idx] = vec4(0.f); rot_d[idx] = vec4(0.f); rot_pg[idx] = vec4(0.f);
      sc_m[idx] = vec3(0.f); sc_v[idx] = vec3(0.f); sc_d[idx] = vec3(0.f); sc_pg[idx] = vec3(0.f);
      sh0_m[idx] = vec3(0.f); sh0_v[idx] = vec3(0.f); sh0_d[idx] = vec3(0.f); sh0_pg[idx] = vec3(0.f);
      for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
        const uint32_t base = idx * (kMaxSphericalHarmonicsCoefficients - 1) + i;
        shrest_m[base] = vec3(0.f);
        shrest_v[base] = vec3(0.f);
        shrest_d[base] = vec3(0.f);
        shrest_pg[base] = vec3(0.f);
      }
    }
  );
}

void Adan::reset_opacity() {
  thrust::fill(m_opacities_m.begin(), m_opacities_m.end(), 0.f);
  thrust::fill(m_opacities_v.begin(), m_opacities_v.end(), 0.f);
  thrust::fill(m_opacities_d.begin(), m_opacities_d.end(), 0.f);
  thrust::fill(m_opacities_prev_g.begin(), m_opacities_prev_g.end(), 0.f);
}

void Adan::remove(char* kept_flag, int num_kept) {
  size_t original_size = m_gaussians->size();
  thrust::device_vector<uint> mapping(original_size);
  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<uint>(0), thrust::make_counting_iterator<uint>(original_size),
    mapping.begin(), [kept_flag] __device__ (uint orig) { return static_cast<bool>(kept_flag[orig]); }
  );

  const uint32_t num_sh_rest = kMaxSphericalHarmonicsCoefficients - 1;

  thrust::device_vector<vec3> means_m(num_kept), means_v(num_kept), means_d(num_kept), means_pg(num_kept);
  thrust::device_vector<float> op_m(num_kept), op_v(num_kept), op_d(num_kept), op_pg(num_kept);
  thrust::device_vector<vec4> rot_m(num_kept), rot_v(num_kept), rot_d(num_kept), rot_pg(num_kept);
  thrust::device_vector<vec3> sc_m(num_kept), sc_v(num_kept), sc_d(num_kept), sc_pg(num_kept);
  thrust::device_vector<vec3> sh0_m(num_kept), sh0_v(num_kept), sh0_d(num_kept), sh0_pg(num_kept);
  thrust::device_vector<vec3> shrest_m(num_kept * num_sh_rest), shrest_v(num_kept * num_sh_rest), shrest_d(num_kept * num_sh_rest), shrest_pg(num_kept * num_sh_rest);

  const int grid = (num_kept + 255) / 256;
  copy_adan_state<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_means_m.data()), thrust::raw_pointer_cast(m_means_v.data()), thrust::raw_pointer_cast(m_means_d.data()), thrust::raw_pointer_cast(m_means_prev_g.data()),
    thrust::raw_pointer_cast(means_m.data()), thrust::raw_pointer_cast(means_v.data()), thrust::raw_pointer_cast(means_d.data()), thrust::raw_pointer_cast(means_pg.data()),
    thrust::raw_pointer_cast(m_opacities_m.data()), thrust::raw_pointer_cast(m_opacities_v.data()), thrust::raw_pointer_cast(m_opacities_d.data()), thrust::raw_pointer_cast(m_opacities_prev_g.data()),
    thrust::raw_pointer_cast(op_m.data()), thrust::raw_pointer_cast(op_v.data()), thrust::raw_pointer_cast(op_d.data()), thrust::raw_pointer_cast(op_pg.data()),
    thrust::raw_pointer_cast(m_rotations_m.data()), thrust::raw_pointer_cast(m_rotations_v.data()), thrust::raw_pointer_cast(m_rotations_d.data()), thrust::raw_pointer_cast(m_rotations_prev_g.data()),
    thrust::raw_pointer_cast(rot_m.data()), thrust::raw_pointer_cast(rot_v.data()), thrust::raw_pointer_cast(rot_d.data()), thrust::raw_pointer_cast(rot_pg.data()),
    thrust::raw_pointer_cast(m_scales_m.data()), thrust::raw_pointer_cast(m_scales_v.data()), thrust::raw_pointer_cast(m_scales_d.data()), thrust::raw_pointer_cast(m_scales_prev_g.data()),
    thrust::raw_pointer_cast(sc_m.data()), thrust::raw_pointer_cast(sc_v.data()), thrust::raw_pointer_cast(sc_d.data()), thrust::raw_pointer_cast(sc_pg.data()),
    thrust::raw_pointer_cast(m_sh0_m.data()), thrust::raw_pointer_cast(m_sh0_v.data()), thrust::raw_pointer_cast(m_sh0_d.data()), thrust::raw_pointer_cast(m_sh0_prev_g.data()),
    thrust::raw_pointer_cast(sh0_m.data()), thrust::raw_pointer_cast(sh0_v.data()), thrust::raw_pointer_cast(sh0_d.data()), thrust::raw_pointer_cast(sh0_pg.data()),
    thrust::raw_pointer_cast(m_shrest_m.data()), thrust::raw_pointer_cast(m_shrest_v.data()), thrust::raw_pointer_cast(m_shrest_d.data()), thrust::raw_pointer_cast(m_shrest_prev_g.data()),
    thrust::raw_pointer_cast(shrest_m.data()), thrust::raw_pointer_cast(shrest_v.data()), thrust::raw_pointer_cast(shrest_d.data()), thrust::raw_pointer_cast(shrest_pg.data()),
    thrust::raw_pointer_cast(mapping.data()),
    num_kept,
    num_sh_rest
  );

  m_means_m = std::move(means_m);
  m_means_v = std::move(means_v);
  m_means_d = std::move(means_d);
  m_means_prev_g = std::move(means_pg);

  m_opacities_m = std::move(op_m);
  m_opacities_v = std::move(op_v);
  m_opacities_d = std::move(op_d);
  m_opacities_prev_g = std::move(op_pg);

  m_rotations_m = std::move(rot_m);
  m_rotations_v = std::move(rot_v);
  m_rotations_d = std::move(rot_d);
  m_rotations_prev_g = std::move(rot_pg);

  m_scales_m = std::move(sc_m);
  m_scales_v = std::move(sc_v);
  m_scales_d = std::move(sc_d);
  m_scales_prev_g = std::move(sc_pg);

  m_sh0_m = std::move(sh0_m);
  m_sh0_v = std::move(sh0_v);
  m_sh0_d = std::move(sh0_d);
  m_sh0_prev_g = std::move(sh0_pg);

  m_shrest_m = std::move(shrest_m);
  m_shrest_v = std::move(shrest_v);
  m_shrest_d = std::move(shrest_d);
  m_shrest_prev_g = std::move(shrest_pg);
}

void Adan::duplicate(int* /*indices*/, int* /*new_indices*/, int /*num_duplicate*/) {
  if (m_gaussians->size() == 0) return;
  const uint32_t n = m_gaussians->size();
  const uint32_t num_sh_rest = kMaxSphericalHarmonicsCoefficients - 1;
  m_means_m.resize(n, vec3(0.f));
  m_means_v.resize(n, vec3(0.f));
  m_means_d.resize(n, vec3(0.f));
  m_means_prev_g.resize(n, vec3(0.f));

  m_opacities_m.resize(n, 0.f);
  m_opacities_v.resize(n, 0.f);
  m_opacities_d.resize(n, 0.f);
  m_opacities_prev_g.resize(n, 0.f);

  m_rotations_m.resize(n, vec4(0.f));
  m_rotations_v.resize(n, vec4(0.f));
  m_rotations_d.resize(n, vec4(0.f));
  m_rotations_prev_g.resize(n, vec4(0.f));

  m_scales_m.resize(n, vec3(0.f));
  m_scales_v.resize(n, vec3(0.f));
  m_scales_d.resize(n, vec3(0.f));
  m_scales_prev_g.resize(n, vec3(0.f));

  m_sh0_m.resize(n, vec3(0.f));
  m_sh0_v.resize(n, vec3(0.f));
  m_sh0_d.resize(n, vec3(0.f));
  m_sh0_prev_g.resize(n, vec3(0.f));

  m_shrest_m.resize(n * num_sh_rest, vec3(0.f));
  m_shrest_v.resize(n * num_sh_rest, vec3(0.f));
  m_shrest_d.resize(n * num_sh_rest, vec3(0.f));
  m_shrest_prev_g.resize(n * num_sh_rest, vec3(0.f));
}

__global__ void reorder_state_kernel(
  const vec3* __restrict__ s_means_m, const vec3* __restrict__ s_means_v, const vec3* __restrict__ s_means_d, const vec3* __restrict__ s_means_pg,
  vec3* __restrict__ d_means_m, vec3* __restrict__ d_means_v, vec3* __restrict__ d_means_d, vec3* __restrict__ d_means_pg,
  const float* __restrict__ s_op_m, const float* __restrict__ s_op_v, const float* __restrict__ s_op_d, const float* __restrict__ s_op_pg,
  float* __restrict__ d_op_m, float* __restrict__ d_op_v, float* __restrict__ d_op_d, float* __restrict__ d_op_pg,
  const vec4* __restrict__ s_rot_m, const vec4* __restrict__ s_rot_v, const vec4* __restrict__ s_rot_d, const vec4* __restrict__ s_rot_pg,
  vec4* __restrict__ d_rot_m, vec4* __restrict__ d_rot_v, vec4* __restrict__ d_rot_d, vec4* __restrict__ d_rot_pg,
  const vec3* __restrict__ s_sc_m, const vec3* __restrict__ s_sc_v, const vec3* __restrict__ s_sc_d, const vec3* __restrict__ s_sc_pg,
  vec3* __restrict__ d_sc_m, vec3* __restrict__ d_sc_v, vec3* __restrict__ d_sc_d, vec3* __restrict__ d_sc_pg,
  const vec3* __restrict__ s_sh0_m, const vec3* __restrict__ s_sh0_v, const vec3* __restrict__ s_sh0_d, const vec3* __restrict__ s_sh0_pg,
  vec3* __restrict__ d_sh0_m, vec3* __restrict__ d_sh0_v, vec3* __restrict__ d_sh0_d, vec3* __restrict__ d_sh0_pg,
  const vec3* __restrict__ s_shrest_m, const vec3* __restrict__ s_shrest_v, const vec3* __restrict__ s_shrest_d, const vec3* __restrict__ s_shrest_pg,
  vec3* __restrict__ d_shrest_m, vec3* __restrict__ d_shrest_v, vec3* __restrict__ d_shrest_d, vec3* __restrict__ d_shrest_pg,
  const uint* __restrict__ mapping,
  int num_items,
  uint32_t num_sh_rest
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;
  uint src_idx = mapping[idx];
  d_means_m[idx] = s_means_m[src_idx]; d_means_v[idx] = s_means_v[src_idx]; d_means_d[idx] = s_means_d[src_idx]; d_means_pg[idx] = s_means_pg[src_idx];
  d_op_m[idx] = s_op_m[src_idx]; d_op_v[idx] = s_op_v[src_idx]; d_op_d[idx] = s_op_d[src_idx]; d_op_pg[idx] = s_op_pg[src_idx];
  d_rot_m[idx] = s_rot_m[src_idx]; d_rot_v[idx] = s_rot_v[src_idx]; d_rot_d[idx] = s_rot_d[src_idx]; d_rot_pg[idx] = s_rot_pg[src_idx];
  d_sc_m[idx] = s_sc_m[src_idx]; d_sc_v[idx] = s_sc_v[src_idx]; d_sc_d[idx] = s_sc_d[src_idx]; d_sc_pg[idx] = s_sc_pg[src_idx];
  d_sh0_m[idx] = s_sh0_m[src_idx]; d_sh0_v[idx] = s_sh0_v[src_idx]; d_sh0_d[idx] = s_sh0_d[src_idx]; d_sh0_pg[idx] = s_sh0_pg[src_idx];
  const int s_rest_start = src_idx * num_sh_rest;
  const int d_rest_start = idx * num_sh_rest;
  for (uint32_t i = 0; i < num_sh_rest; i++) {
    d_shrest_m[d_rest_start + i] = s_shrest_m[s_rest_start + i];
    d_shrest_v[d_rest_start + i] = s_shrest_v[s_rest_start + i];
    d_shrest_d[d_rest_start + i] = s_shrest_d[s_rest_start + i];
    d_shrest_pg[d_rest_start + i] = s_shrest_pg[s_rest_start + i];
  }
}

void Adan::reorder(uint* indices) {
  int num_gaussians = (int)m_means_m.size();
  const uint32_t num_sh_rest = kMaxSphericalHarmonicsCoefficients - 1;

  thrust::device_vector<vec3> means_m(num_gaussians), means_v(num_gaussians), means_d(num_gaussians), means_pg(num_gaussians);
  thrust::device_vector<float> op_m(num_gaussians), op_v(num_gaussians), op_d(num_gaussians), op_pg(num_gaussians);
  thrust::device_vector<vec4> rot_m(num_gaussians), rot_v(num_gaussians), rot_d(num_gaussians), rot_pg(num_gaussians);
  thrust::device_vector<vec3> sc_m(num_gaussians), sc_v(num_gaussians), sc_d(num_gaussians), sc_pg(num_gaussians);
  thrust::device_vector<vec3> sh0_m(num_gaussians), sh0_v(num_gaussians), sh0_d(num_gaussians), sh0_pg(num_gaussians);
  thrust::device_vector<vec3> shrest_m(num_gaussians * num_sh_rest), shrest_v(num_gaussians * num_sh_rest), shrest_d(num_gaussians * num_sh_rest), shrest_pg(num_gaussians * num_sh_rest);

  const int grid = (num_gaussians + 255) / 256;
  reorder_state_kernel<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_means_m.data()), thrust::raw_pointer_cast(m_means_v.data()), thrust::raw_pointer_cast(m_means_d.data()), thrust::raw_pointer_cast(m_means_prev_g.data()),
    thrust::raw_pointer_cast(means_m.data()), thrust::raw_pointer_cast(means_v.data()), thrust::raw_pointer_cast(means_d.data()), thrust::raw_pointer_cast(means_pg.data()),
    thrust::raw_pointer_cast(m_opacities_m.data()), thrust::raw_pointer_cast(m_opacities_v.data()), thrust::raw_pointer_cast(m_opacities_d.data()), thrust::raw_pointer_cast(m_opacities_prev_g.data()),
    thrust::raw_pointer_cast(op_m.data()), thrust::raw_pointer_cast(op_v.data()), thrust::raw_pointer_cast(op_d.data()), thrust::raw_pointer_cast(op_pg.data()),
    thrust::raw_pointer_cast(m_rotations_m.data()), thrust::raw_pointer_cast(m_rotations_v.data()), thrust::raw_pointer_cast(m_rotations_d.data()), thrust::raw_pointer_cast(m_rotations_prev_g.data()),
    thrust::raw_pointer_cast(rot_m.data()), thrust::raw_pointer_cast(rot_v.data()), thrust::raw_pointer_cast(rot_d.data()), thrust::raw_pointer_cast(rot_pg.data()),
    thrust::raw_pointer_cast(m_scales_m.data()), thrust::raw_pointer_cast(m_scales_v.data()), thrust::raw_pointer_cast(m_scales_d.data()), thrust::raw_pointer_cast(m_scales_prev_g.data()),
    thrust::raw_pointer_cast(sc_m.data()), thrust::raw_pointer_cast(sc_v.data()), thrust::raw_pointer_cast(sc_d.data()), thrust::raw_pointer_cast(sc_pg.data()),
    thrust::raw_pointer_cast(m_sh0_m.data()), thrust::raw_pointer_cast(m_sh0_v.data()), thrust::raw_pointer_cast(m_sh0_d.data()), thrust::raw_pointer_cast(m_sh0_prev_g.data()),
    thrust::raw_pointer_cast(sh0_m.data()), thrust::raw_pointer_cast(sh0_v.data()), thrust::raw_pointer_cast(sh0_d.data()), thrust::raw_pointer_cast(sh0_pg.data()),
    thrust::raw_pointer_cast(m_shrest_m.data()), thrust::raw_pointer_cast(m_shrest_v.data()), thrust::raw_pointer_cast(m_shrest_d.data()), thrust::raw_pointer_cast(m_shrest_prev_g.data()),
    thrust::raw_pointer_cast(shrest_m.data()), thrust::raw_pointer_cast(shrest_v.data()), thrust::raw_pointer_cast(shrest_d.data()), thrust::raw_pointer_cast(shrest_pg.data()),
    indices,
    num_gaussians,
    num_sh_rest
  );

  m_means_m = std::move(means_m);
  m_means_v = std::move(means_v);
  m_means_d = std::move(means_d);
  m_means_prev_g = std::move(means_pg);
  m_opacities_m = std::move(op_m);
  m_opacities_v = std::move(op_v);
  m_opacities_d = std::move(op_d);
  m_opacities_prev_g = std::move(op_pg);
  m_rotations_m = std::move(rot_m);
  m_rotations_v = std::move(rot_v);
  m_rotations_d = std::move(rot_d);
  m_rotations_prev_g = std::move(rot_pg);
  m_scales_m = std::move(sc_m);
  m_scales_v = std::move(sc_v);
  m_scales_d = std::move(sc_d);
  m_scales_prev_g = std::move(sc_pg);
  m_sh0_m = std::move(sh0_m);
  m_sh0_v = std::move(sh0_v);
  m_sh0_d = std::move(sh0_d);
  m_sh0_prev_g = std::move(sh0_pg);
  m_shrest_m = std::move(shrest_m);
  m_shrest_v = std::move(shrest_v);
  m_shrest_d = std::move(shrest_d);
  m_shrest_prev_g = std::move(shrest_pg);
}

void Adan::set_params(const json& config) {
  OptimizerBase::set_params(config);
  m_adan_params.from_json(config);
}

json Adan::get_params() const {
  json params = OptimizerBase::get_params();
  params["type"] = "adan";
  json ap = m_adan_params.to_json();
  for (auto it = ap.begin(); it != ap.end(); ++it) {
    params[it.key()] = it.value();
  }
  return params;
}

} // namespace tinygs