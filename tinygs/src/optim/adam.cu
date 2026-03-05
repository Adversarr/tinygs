#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <nvtx3/nvtx3.hpp>
#include <cmath>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adam.hpp"

#include "../helper_math.h"

constexpr int block_size = 512; // make occupancy higher

namespace tinygs {

__device__ static __forceinline__ float lerp(float v0, float v1, float t) {
    return fmaf(t, v1, fmaf(-t, v0, v0));
}

enum AdamL1DecayMode : int {
  kAdamL1DecayNone = 0,
  kAdamL1DecayOpacity = 1,
  kAdamL1DecayScale = 2,
};

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
    float l1_decay,
    int l1_decay_mode
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians) return;

  // Load
  float theta = thetas[idx];
  float g = gradient_scale * thetas_grad[idx];
  if (l1_decay != 0.0f) {
    if (l1_decay_mode == kAdamL1DecayOpacity) {
      g += l1_decay * activate_opacity_deriv(theta);
    } else if (l1_decay_mode == kAdamL1DecayScale) {
      g += l1_decay * activate_scale_deriv(theta);
    }
  }
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
  const float denom = adam_p.tf_style
      ? (tinygs::sqrt(v) + adam_p.epsilon) / bias_correction2_sqrt
      : tinygs::sqrt(v) / bias_correction2_sqrt + adam_p.epsilon;

  // step
  theta -= m_hat * lr / denom;

  // write back updated params and moments
  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}


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
    float max_grad_1
) {

  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians) return;

  // Load
  float theta = thetas[idx];
  theta -= lr * adam_p.weight_decay * theta;
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
  const float denom = adam_p.tf_style
      ? (tinygs::sqrt(v) + adam_p.epsilon) / bias_correction2_sqrt
      : tinygs::sqrt(v) / bias_correction2_sqrt + adam_p.epsilon;

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



void Adam::step(float scale, BackendStream stream) {
  if (m_adam_params.decouple_decay) {
    step_adamw(scale, stream);
  } else {
    step_adam(scale, stream);
  }
}

void Adam::step(const GroupStepConfig& step_config, BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  if (!step_config.any_update()) {
    return;
  }
  if (m_adam_params.decouple_decay) {
    if (!(step_config.update_means && step_config.update_shs && step_config.update_opacities &&
          step_config.update_scales && step_config.update_rotations)) {
      throw std::runtime_error("Adam(decouple_decay=true) does not support selective group stepping.");
    }
    const float s = step_config.means_scale;
    const float tol = 1e-7f;
    if (std::fabs(step_config.shs_scale - s) > tol || std::fabs(step_config.opacities_scale - s) > tol ||
        std::fabs(step_config.scales_scale - s) > tol || std::fabs(step_config.rotations_scale - s) > tol) {
      throw std::runtime_error("Adam(decouple_decay=true) requires equal group scales.");
    }
    step_adamw(s, stream);
    return;
  }

  constexpr int block_size = 256;
  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("Adam::step: gaussians or gaussians_grad is null");
  }
  if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("Adam::step: gaussians and gaussians_grad must have same size");
  }

  const auto n = m_gaussians->size();
  const float g_scale = 1.0f / n;
  const float scene_scale = m_gaussians->scene_scale();
  m_global_steps++;

  if (step_config.update_means) {
    m_means_steps++;
    const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                                                         static_cast<double>(m_means_steps)));
    const float bc2 = static_cast<float>(std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                                                                   static_cast<double>(m_means_steps))));
    adam<<<div_round_up<uint>(n * 3, block_size), block_size, 0, cuda_stream>>>(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
      reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->means().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_second.data())),
      m_adam_params,
      m_params.means_lr * scene_scale * m_means_global_lr,
      n * 3,
      step_config.means_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      0.0f,
      kAdamL1DecayNone);
  }

  if (step_config.update_opacities) {
    m_opacities_steps++;
    const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                                                         static_cast<double>(m_opacities_steps)));
    const float bc2 = static_cast<float>(std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                                                                   static_cast<double>(m_opacities_steps))));
    adam<<<div_round_up<uint>(n, block_size), block_size, 0, cuda_stream>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      (const float*) thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_opacities_first.data()),
      (float*) thrust::raw_pointer_cast(m_opacities_second.data()),
      m_adam_params,
      m_params.opacities_lr * m_opacities_global_lr,
      n,
      step_config.opacities_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      m_params.opacities_l1 * g_scale,
      kAdamL1DecayOpacity);
  }

  if (step_config.update_rotations) {
    m_rotations_steps++;
    const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                                                         static_cast<double>(m_rotations_steps)));
    const float bc2 = static_cast<float>(std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                                                                   static_cast<double>(m_rotations_steps))));
    adam<<<div_round_up<uint>(n * 4, block_size), block_size, 0, cuda_stream>>>(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
      reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->rotations().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_second.data())),
      m_adam_params,
      m_params.rotations_lr * m_rotations_global_lr,
      n * 4,
      step_config.rotations_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      0.0f,
      kAdamL1DecayNone);
  }

  if (step_config.update_scales) {
    m_scales_steps++;
    const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                                                         static_cast<double>(m_scales_steps)));
    const float bc2 = static_cast<float>(std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                                                                   static_cast<double>(m_scales_steps))));
    adam<<<div_round_up<uint>(n * 3, block_size), block_size, 0, cuda_stream>>>(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
      reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->scales().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_second.data())),
      m_adam_params,
      m_params.scales_lr * m_scales_global_lr,
      n * 3,
      step_config.scales_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      m_params.scales_l1 * g_scale,
      kAdamL1DecayScale);
  }

  if (step_config.update_shs) {
    m_shs_steps++;
    const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                                                         static_cast<double>(m_shs_steps)));
    const float bc2 = static_cast<float>(std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                                                                   static_cast<double>(m_shs_steps))));
    adam<<<div_round_up<uint>(n * 3, block_size), block_size, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh0().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh0().data()),
      thrust::raw_pointer_cast(m_sh0_first.data()),
      thrust::raw_pointer_cast(m_sh0_second.data()),
      m_adam_params,
      m_params.shs_lr * m_shs_global_lr,
      n * 3,
      step_config.shs_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      0.0f,
      kAdamL1DecayNone);
    adam<<<div_round_up<uint>(n * 9, block_size), block_size, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh1().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh1().data()),
      thrust::raw_pointer_cast(m_sh1_first.data()),
      thrust::raw_pointer_cast(m_sh1_second.data()),
      m_adam_params,
      m_params.shs_lr * m_params.sh1_lr_scale * m_shs_global_lr,
      n * 9,
      step_config.shs_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      0.0f,
      kAdamL1DecayNone);
    adam<<<div_round_up<uint>(n * 15, block_size), block_size, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh2().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh2().data()),
      thrust::raw_pointer_cast(m_sh2_first.data()),
      thrust::raw_pointer_cast(m_sh2_second.data()),
      m_adam_params,
      m_params.shs_lr * m_params.sh2_lr_scale * m_shs_global_lr,
      n * 15,
      step_config.shs_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      0.0f,
      kAdamL1DecayNone);
    adam<<<div_round_up<uint>(n * 21, block_size), block_size, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh3().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh3().data()),
      thrust::raw_pointer_cast(m_sh3_first.data()),
      thrust::raw_pointer_cast(m_sh3_second.data()),
      m_adam_params,
      m_params.shs_lr * m_params.sh3_lr_scale * m_shs_global_lr,
      n * 21,
      step_config.shs_scale,
      bc1,
      bc2,
      m_params.max_grad_1,
      0.0f,
      kAdamL1DecayNone);
  }

  maybe_sync(stream);
}


void Adam::step_adam(float scale, BackendStream stream) {
  GroupStepConfig step_config;
  step_config.update_means = true;
  step_config.update_shs = true;
  step_config.update_opacities = true;
  step_config.update_scales = true;
  step_config.update_rotations = true;
  step_config.means_scale = scale;
  step_config.shs_scale = scale;
  step_config.opacities_scale = scale;
  step_config.scales_scale = scale;
  step_config.rotations_scale = scale;
  step(step_config, stream);
}


void Adam::step_adamw(float scale, BackendStream stream) {
  NVTX3_FUNC_RANGE();
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  const float gradient_scale = scale;
  constexpr int block_size = 256;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("Adam::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("Adam::step: gaussians and gaussians_grad must have same size");
  }

  auto n = m_gaussians->size();
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

    // Means (3 floats per Gaussian)
    adamw<<<div_round_up<uint>(n * 3, block_size), block_size, 0, cuda_stream>>>(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
      reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->means().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_second.data())),
      m_adam_params,
      m_params.means_lr * scene_scale * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1);

    // Opacities (1 float per Gaussian)
    adamw<<<div_round_up<uint>(n, block_size), block_size, 0, cuda_stream>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      (const float*) thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_opacities_first.data()),
      (float*) thrust::raw_pointer_cast(m_opacities_second.data()),
      m_adam_params,
      m_params.opacities_lr * m_global_lr,
      n,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1);

    // Rotations (4 floats per Gaussian)
    adamw<<<div_round_up<uint>(n * 4, block_size), block_size, 0, cuda_stream>>>(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
      reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->rotations().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_second.data())),
      m_adam_params,
      m_params.rotations_lr * m_global_lr,
      n * 4,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1);

    // Scales (3 floats per Gaussian)
    adamw<<<div_round_up<uint>(n * 3, block_size), block_size, 0, cuda_stream>>>(
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
      reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->scales().data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_first.data())),
      reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_second.data())),
      m_adam_params,
      m_params.scales_lr * m_global_lr,
      n * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1);

    // SH degree 0 (1 coefficient, 3*N floats)
    adamw<<<div_round_up<uint>(n * 3, block_size), block_size, 0, cuda_stream>>>(
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
      m_params.max_grad_1);

    // SH degree 1 (3 coefficients, 9*N floats)
    adamw<<<div_round_up<uint>(n * 9, block_size), block_size, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh1().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh1().data()),
      thrust::raw_pointer_cast(m_sh1_first.data()),
      thrust::raw_pointer_cast(m_sh1_second.data()),
      m_adam_params,
      m_params.shs_lr * m_params.sh1_lr_scale * m_global_lr,
      n * 9,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1);

    // SH degree 2 (5 coefficients, 15*N floats)
    adamw<<<div_round_up<uint>(n * 15, block_size), block_size, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh2().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh2().data()),
      thrust::raw_pointer_cast(m_sh2_first.data()),
      thrust::raw_pointer_cast(m_sh2_second.data()),
      m_adam_params,
      m_params.shs_lr * m_params.sh2_lr_scale * m_global_lr,
      n * 15,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1);

    // SH degree 3 (7 coefficients, 21*N floats)
    adamw<<<div_round_up<uint>(n * 21, block_size), block_size, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh3().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh3().data()),
      thrust::raw_pointer_cast(m_sh3_first.data()),
      thrust::raw_pointer_cast(m_sh3_second.data()),
      m_adam_params,
      m_params.shs_lr * m_params.sh3_lr_scale * m_global_lr,
      n * 21,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt,
      m_params.max_grad_1);

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
  m_means_steps = 0;
  m_shs_steps = 0;
  m_opacities_steps = 0;
  m_scales_steps = 0;
  m_rotations_steps = 0;

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
  j["decouple_decay"] = decouple_decay;
  j["weight_decay"] = weight_decay;
  j["tf_style"] = tf_style;
  return j;
}

void AdamParameters::from_json(const json& config) {
  if (config.contains("beta1")) beta1 = config.at("beta1").get<float>();
  if (config.contains("beta2")) beta2 = config.at("beta2").get<float>();
  if (config.contains("epsilon")) epsilon = config.at("epsilon").get<float>();
  if (config.contains("decouple_decay")) decouple_decay = config.at("decouple_decay").get<bool>();
  if (config.contains("weight_decay")) weight_decay = config.at("weight_decay").get<float>();
  if (config.contains("tf_style")) tf_style = config.at("tf_style").get<bool>();
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
