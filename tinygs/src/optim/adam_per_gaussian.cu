#include <nvtx3/nvtx3.hpp>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

#include <cmath>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adam_per_gaussian.hpp"

#include "../helper_math.h"

constexpr int block_size = 512;

namespace tinygs {

__device__ static __forceinline__ float lerp(float v0, float v1, float t) {
  return fmaf(t, v1, fmaf(-t, v0, v0));
}

enum AdamL1DecayMode : int {
  kAdamL1DecayNone = 0,
  kAdamL1DecayOpacity = 1,
  kAdamL1DecayScale = 2,
};

enum GaussianIndexMode : int {
  kGaussianIndexDiv = 0,
  kGaussianIndexMod = 1,
};

__device__ static __forceinline__ uint32_t gaussian_index(uint32_t idx,
                                                           uint32_t n,
                                                           uint32_t channels_per_gaussian,
                                                           int index_mode) {
  if (index_mode == kGaussianIndexMod) {
    return idx % n;
  }
  return idx / channels_per_gaussian;
}

__global__ static void increment_steps(uint32_t* __restrict__ steps, uint32_t n) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  steps[idx] += 1;
}

__global__ static void adam_pg(
    float* __restrict__ thetas,
    float const* __restrict__ thetas_grad,
    float* __restrict__ thetas_first,
    float* __restrict__ thetas_second,
    const uint32_t* __restrict__ steps,
    AdamParameters adam_p,
    float lr,
    uint32_t num_params,
    uint32_t num_gaussians,
    uint32_t channels_per_gaussian,
    int gaussian_index_mode,
    float gradient_scale,
    float max_grad_1,
    float l1_decay,
    int l1_decay_mode) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_params) return;

  const uint32_t gidx = gaussian_index(idx, num_gaussians, channels_per_gaussian, gaussian_index_mode);
  const uint32_t t = max(steps[gidx], 1u);
  const float bias_correction1 = 1.0f - powf(adam_p.beta1, static_cast<float>(t));
  const float bias_correction2_sqrt = sqrtf(1.0f - powf(adam_p.beta2, static_cast<float>(t)));

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

  m = lerp(m, g, 1.0f - adam_p.beta1);
  v = lerp(v, g_sq, 1.0f - adam_p.beta2);

  const float m_hat = m / bias_correction1;
  const float denom = adam_p.tf_style
                          ? (tinygs::sqrt(v) + adam_p.epsilon) / bias_correction2_sqrt
                          : tinygs::sqrt(v) / bias_correction2_sqrt + adam_p.epsilon;

  theta -= m_hat * lr / denom;

  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}

__global__ static void adamw_pg(
    float* __restrict__ thetas,
    float const* __restrict__ thetas_grad,
    float* __restrict__ thetas_first,
    float* __restrict__ thetas_second,
    const uint32_t* __restrict__ steps,
    AdamParameters adam_p,
    float lr,
    uint32_t num_params,
    uint32_t num_gaussians,
    uint32_t channels_per_gaussian,
    int gaussian_index_mode,
    float gradient_scale,
    float max_grad_1) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_params) return;

  const uint32_t gidx = gaussian_index(idx, num_gaussians, channels_per_gaussian, gaussian_index_mode);
  const uint32_t t = max(steps[gidx], 1u);
  const float bias_correction1 = 1.0f - powf(adam_p.beta1, static_cast<float>(t));
  const float bias_correction2_sqrt = sqrtf(1.0f - powf(adam_p.beta2, static_cast<float>(t)));

  float theta = thetas[idx];
  theta -= lr * adam_p.weight_decay * theta;

  float g = gradient_scale * thetas_grad[idx];
  if (max_grad_1 != 0.0f) {
    g = copysignf(fminf(fabsf(g), max_grad_1), g);
  }
  float m = thetas_first[idx];
  float v = thetas_second[idx];
  const float g_sq = g * g;

  m = lerp(m, g, 1.0f - adam_p.beta1);
  v = lerp(v, g_sq, 1.0f - adam_p.beta2);

  const float m_hat = m / bias_correction1;
  const float denom = adam_p.tf_style
                          ? (tinygs::sqrt(v) + adam_p.epsilon) / bias_correction2_sqrt
                          : tinygs::sqrt(v) / bias_correction2_sqrt + adam_p.epsilon;

  theta -= m_hat * lr / denom;

  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}

struct AdamPerGaussian_domain {
  static constexpr char const* name{"optim"};
};
using range = nvtx3::scoped_range_in<AdamPerGaussian_domain>;
using regstr = nvtx3::registered_string_in<AdamPerGaussian_domain>;
struct m_step {
  static constexpr char const* message{"adam_per_gaussian_step"};
};

void AdamPerGaussian::step(float scale, cudaStream_t stream) {
  if (m_adam_params.decouple_decay) {
    step_adamw(scale, stream);
  } else {
    step_adam(scale, stream);
  }
}

void AdamPerGaussian::step(const GroupStepConfig& step_config, cudaStream_t stream) {
  if (!step_config.any_update()) {
    return;
  }
  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("AdamPerGaussian::step: gaussians or gaussians_grad is null");
  }
  if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("AdamPerGaussian::step: gaussians and gaussians_grad must have same size");
  }

  constexpr int launch_block_size = 256;
  const auto n = static_cast<uint32_t>(m_gaussians->size());
  const float g_scale = n > 0 ? (1.0f / static_cast<float>(n)) : 1.0f;
  const float scene_scale = m_gaussians->scene_scale();

  increment_steps<<<div_round_up<uint>(n, launch_block_size), launch_block_size, 0, stream>>>(
      thrust::raw_pointer_cast(m_steps.data()), n);

  const auto* steps_ptr = thrust::raw_pointer_cast(m_steps.data());
  const bool use_adamw = m_adam_params.decouple_decay;

  if (step_config.update_means) {
    if (use_adamw) {
      adamw_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, stream>>>(
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
          reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->means().data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_first.data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_second.data())),
          steps_ptr,
          m_adam_params,
          m_params.means_lr * scene_scale * m_means_global_lr,
          n * 3,
          n,
          3,
          kGaussianIndexDiv,
          step_config.means_scale,
          m_params.max_grad_1);
    } else {
      adam_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, stream>>>(
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->means().data())),
          reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->means().data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_first.data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_means_second.data())),
          steps_ptr,
          m_adam_params,
          m_params.means_lr * scene_scale * m_means_global_lr,
          n * 3,
          n,
          3,
          kGaussianIndexDiv,
          step_config.means_scale,
          m_params.max_grad_1,
          0.0f,
          kAdamL1DecayNone);
    }
  }

  if (step_config.update_opacities) {
    if (use_adamw) {
      adamw_pg<<<div_round_up<uint>(n, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->opacities().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
          thrust::raw_pointer_cast(m_opacities_first.data()),
          thrust::raw_pointer_cast(m_opacities_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.opacities_lr * m_opacities_global_lr,
          n,
          n,
          1,
          kGaussianIndexDiv,
          step_config.opacities_scale,
          m_params.max_grad_1);
    } else {
      adam_pg<<<div_round_up<uint>(n, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->opacities().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
          thrust::raw_pointer_cast(m_opacities_first.data()),
          thrust::raw_pointer_cast(m_opacities_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.opacities_lr * m_opacities_global_lr,
          n,
          n,
          1,
          kGaussianIndexDiv,
          step_config.opacities_scale,
          m_params.max_grad_1,
          m_params.opacities_l1 * g_scale,
          kAdamL1DecayOpacity);
    }
  }

  if (step_config.update_rotations) {
    if (use_adamw) {
      adamw_pg<<<div_round_up<uint>(n * 4, launch_block_size), launch_block_size, 0, stream>>>(
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
          reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->rotations().data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_first.data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_second.data())),
          steps_ptr,
          m_adam_params,
          m_params.rotations_lr * m_rotations_global_lr,
          n * 4,
          n,
          4,
          kGaussianIndexDiv,
          step_config.rotations_scale,
          m_params.max_grad_1);
    } else {
      adam_pg<<<div_round_up<uint>(n * 4, launch_block_size), launch_block_size, 0, stream>>>(
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->rotations().data())),
          reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->rotations().data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_first.data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_rotations_second.data())),
          steps_ptr,
          m_adam_params,
          m_params.rotations_lr * m_rotations_global_lr,
          n * 4,
          n,
          4,
          kGaussianIndexDiv,
          step_config.rotations_scale,
          m_params.max_grad_1,
          0.0f,
          kAdamL1DecayNone);
    }
  }

  if (step_config.update_scales) {
    if (use_adamw) {
      adamw_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, stream>>>(
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
          reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->scales().data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_first.data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_second.data())),
          steps_ptr,
          m_adam_params,
          m_params.scales_lr * m_scales_global_lr,
          n * 3,
          n,
          3,
          kGaussianIndexDiv,
          step_config.scales_scale,
          m_params.max_grad_1);
    } else {
      adam_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, stream>>>(
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_gaussians->scales().data())),
          reinterpret_cast<const float*>(thrust::raw_pointer_cast(m_gaussians_grad->scales().data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_first.data())),
          reinterpret_cast<float*>(thrust::raw_pointer_cast(m_scales_second.data())),
          steps_ptr,
          m_adam_params,
          m_params.scales_lr * m_scales_global_lr,
          n * 3,
          n,
          3,
          kGaussianIndexDiv,
          step_config.scales_scale,
          m_params.max_grad_1,
          m_params.scales_l1 * g_scale,
          kAdamL1DecayScale);
    }
  }

  if (step_config.update_shs) {
    if (use_adamw) {
      adamw_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh0().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh0().data()),
          thrust::raw_pointer_cast(m_sh0_first.data()),
          thrust::raw_pointer_cast(m_sh0_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_shs_global_lr,
          n * 3,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1);
      adamw_pg<<<div_round_up<uint>(n * 9, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh1().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh1().data()),
          thrust::raw_pointer_cast(m_sh1_first.data()),
          thrust::raw_pointer_cast(m_sh1_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh1_lr_scale * m_shs_global_lr,
          n * 9,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1);
      adamw_pg<<<div_round_up<uint>(n * 15, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh2().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh2().data()),
          thrust::raw_pointer_cast(m_sh2_first.data()),
          thrust::raw_pointer_cast(m_sh2_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh2_lr_scale * m_shs_global_lr,
          n * 15,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1);
      adamw_pg<<<div_round_up<uint>(n * 21, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh3().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh3().data()),
          thrust::raw_pointer_cast(m_sh3_first.data()),
          thrust::raw_pointer_cast(m_sh3_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh3_lr_scale * m_shs_global_lr,
          n * 21,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1);
    } else {
      adam_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh0().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh0().data()),
          thrust::raw_pointer_cast(m_sh0_first.data()),
          thrust::raw_pointer_cast(m_sh0_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_shs_global_lr,
          n * 3,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1,
          0.0f,
          kAdamL1DecayNone);
      adam_pg<<<div_round_up<uint>(n * 9, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh1().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh1().data()),
          thrust::raw_pointer_cast(m_sh1_first.data()),
          thrust::raw_pointer_cast(m_sh1_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh1_lr_scale * m_shs_global_lr,
          n * 9,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1,
          0.0f,
          kAdamL1DecayNone);
      adam_pg<<<div_round_up<uint>(n * 15, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh2().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh2().data()),
          thrust::raw_pointer_cast(m_sh2_first.data()),
          thrust::raw_pointer_cast(m_sh2_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh2_lr_scale * m_shs_global_lr,
          n * 15,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1,
          0.0f,
          kAdamL1DecayNone);
      adam_pg<<<div_round_up<uint>(n * 21, launch_block_size), launch_block_size, 0, stream>>>(
          thrust::raw_pointer_cast(m_gaussians->sh3().data()),
          thrust::raw_pointer_cast(m_gaussians_grad->sh3().data()),
          thrust::raw_pointer_cast(m_sh3_first.data()),
          thrust::raw_pointer_cast(m_sh3_second.data()),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh3_lr_scale * m_shs_global_lr,
          n * 21,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1,
          0.0f,
          kAdamL1DecayNone);
    }
  }

  maybe_sync(stream);
}

void AdamPerGaussian::step_adam(float scale, cudaStream_t stream) {
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

void AdamPerGaussian::step_adamw(float scale, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
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

AdamPerGaussian::AdamPerGaussian(std::shared_ptr<GPUGaussian3d> gaussians,
                                 std::shared_ptr<GPUGaussian3d> gaussians_grad) :
    OptimizerBase(gaussians, gaussians_grad) {
  AdamPerGaussian::reset();
}

__global__ static void gather_soa_optim(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        const uint* __restrict__ mapping,
                                        int num_items,
                                        int num_channels,
                                        int src_stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_items * num_channels) return;
  int item_idx = tid % num_items;
  int channel = tid / num_items;
  dst[channel * num_items + item_idx] = src[channel * src_stride + mapping[item_idx]];
}

static void gather_soa_optim_buffers(const thrust::device_vector<float>& src_first,
                                     const thrust::device_vector<float>& src_second,
                                     thrust::device_vector<float>& dst_first,
                                     thrust::device_vector<float>& dst_second,
                                     const uint* mapping,
                                     int num_items,
                                     int num_channels,
                                     int src_stride) {
  int total = num_items * num_channels;
  dst_first.resize(total, 0.f);
  dst_second.resize(total, 0.f);
  if (total == 0) return;
  const int grid = (total + block_size - 1) / block_size;
  gather_soa_optim<<<grid, block_size>>>(
      thrust::raw_pointer_cast(src_first.data()),
      thrust::raw_pointer_cast(dst_first.data()),
      mapping,
      num_items,
      num_channels,
      src_stride);
  gather_soa_optim<<<grid, block_size>>>(
      thrust::raw_pointer_cast(src_second.data()),
      thrust::raw_pointer_cast(dst_second.data()),
      mapping,
      num_items,
      num_channels,
      src_stride);
}

static void relayout_soa_optim(thrust::device_vector<float>& buf, int old_n, int new_n, int num_channels) {
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
      old_n,
      num_channels,
      old_n);
  buf = std::move(new_buf);
}

__global__ static void copy_optimizer_base_state_pg(const vec3* __restrict__ src_means_first,
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
                                                     const uint* __restrict__ mapping,
                                                     int num_items) {
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
}

void AdamPerGaussian::remove(char* kept_flag, int num_kept) {
  size_t original_size = m_means_first.size();
  thrust::device_vector<uint> mapping(original_size);

  auto end_it = thrust::copy_if(thrust::device,
                                thrust::make_counting_iterator<uint>(0),
                                thrust::make_counting_iterator<uint>(original_size),
                                mapping.begin(),
                                [kept_flag] __device__(uint orig) { return static_cast<bool>(kept_flag[orig]); });
  const int kept_from_mapping = static_cast<int>(end_it - mapping.begin());
  if (kept_from_mapping != num_kept) {
    throw std::runtime_error("AdamPerGaussian::remove: num_kept mismatch with keep mask");
  }

  thrust::device_vector<vec3> means_first(num_kept);
  thrust::device_vector<vec3> means_second(num_kept);
  thrust::device_vector<float> opacities_first(num_kept);
  thrust::device_vector<float> opacities_second(num_kept);
  thrust::device_vector<vec4> rotations_first(num_kept);
  thrust::device_vector<vec4> rotations_second(num_kept);
  thrust::device_vector<vec3> scales_first(num_kept);
  thrust::device_vector<vec3> scales_second(num_kept);

  const int grid = (num_kept + block_size - 1) / block_size;
  copy_optimizer_base_state_pg<<<grid, block_size>>>(
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
      num_kept);

  m_means_first = std::move(means_first);
  m_means_second = std::move(means_second);
  m_opacities_first = std::move(opacities_first);
  m_opacities_second = std::move(opacities_second);
  m_rotations_first = std::move(rotations_first);
  m_rotations_second = std::move(rotations_second);
  m_scales_first = std::move(scales_first);
  m_scales_second = std::move(scales_second);

  thrust::device_vector<uint32_t> new_steps(num_kept, 0);
  thrust::gather(thrust::device,
                 mapping.begin(),
                 mapping.begin() + num_kept,
                 m_steps.begin(),
                 new_steps.begin());
  m_steps = std::move(new_steps);

  const uint* map_ptr = thrust::raw_pointer_cast(mapping.data());
  const int old_n = static_cast<int>(original_size);
  thrust::device_vector<float> sh0_f, sh0_s, sh1_f, sh1_s, sh2_f, sh2_s, sh3_f, sh3_s;
  gather_soa_optim_buffers(
      m_sh0_first, m_sh0_second, sh0_f, sh0_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(0) * 3, old_n);
  gather_soa_optim_buffers(
      m_sh1_first, m_sh1_second, sh1_f, sh1_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(1) * 3, old_n);
  gather_soa_optim_buffers(
      m_sh2_first, m_sh2_second, sh2_f, sh2_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(2) * 3, old_n);
  gather_soa_optim_buffers(
      m_sh3_first, m_sh3_second, sh3_f, sh3_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(3) * 3, old_n);
  m_sh0_first = std::move(sh0_f);
  m_sh0_second = std::move(sh0_s);
  m_sh1_first = std::move(sh1_f);
  m_sh1_second = std::move(sh1_s);
  m_sh2_first = std::move(sh2_f);
  m_sh2_second = std::move(sh2_s);
  m_sh3_first = std::move(sh3_f);
  m_sh3_second = std::move(sh3_s);
}

void AdamPerGaussian::duplicate(int* indices, int* new_indices, int num_duplicate) {
  if (num_duplicate == 0) return;
  (void)indices;
  (void)new_indices;

  const int old_n = static_cast<int>(m_means_first.size());
  const int new_n = static_cast<int>(m_gaussians->size());

  m_means_first.resize(new_n, vec3(0.f, 0.f, 0.f));
  m_means_second.resize(new_n, vec3(0.f, 0.f, 0.f));
  m_opacities_first.resize(new_n, 0.f);
  m_opacities_second.resize(new_n, 0.f);
  m_rotations_first.resize(new_n, vec4(0.f, 0.f, 0.f, 0.f));
  m_rotations_second.resize(new_n, vec4(0.f, 0.f, 0.f, 0.f));
  m_scales_first.resize(new_n, vec3(0.f, 0.f, 0.f));
  m_scales_second.resize(new_n, vec3(0.f, 0.f, 0.f));
  m_steps.resize(new_n, 0);

  for (int deg = 0; deg < 4; deg++) {
    int nc = GPUGaussian3d::sh_degree_num_coeffs(deg) * 3;
    auto relayout_pair = [&](thrust::device_vector<float>& first, thrust::device_vector<float>& second) {
      relayout_soa_optim(first, old_n, new_n, nc);
      relayout_soa_optim(second, old_n, new_n, nc);
    };
    switch (deg) {
      case 0:
        relayout_pair(m_sh0_first, m_sh0_second);
        break;
      case 1:
        relayout_pair(m_sh1_first, m_sh1_second);
        break;
      case 2:
        relayout_pair(m_sh2_first, m_sh2_second);
        break;
      case 3:
        relayout_pair(m_sh3_first, m_sh3_second);
        break;
    }
  }
}

void AdamPerGaussian::reset() {
  size_t num_gaussians = m_gaussians->size();

  m_steps.assign(num_gaussians, 0);

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

void AdamPerGaussian::reset(int* indices, int num_reset) {
  const int n = static_cast<int>(m_gaussians->size());
  thrust::for_each(
      thrust::device_ptr<int>(indices),
      thrust::device_ptr<int>(indices) + num_reset,
      [means_first = m_means_first.data(),
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
       steps = thrust::raw_pointer_cast(m_steps.data()),
       n] __device__(int idx) {
        means_first[idx] = vec3(0.f);
        means_second[idx] = vec3(0.f);
        opacities_first[idx] = 0.f;
        opacities_second[idx] = 0.f;
        rotations_first[idx] = vec4(0.f);
        rotations_second[idx] = vec4(0.f);
        scales_first[idx] = vec3(0.f);
        scales_second[idx] = vec3(0.f);
        steps[idx] = 0;
        for (int ch = 0; ch < 1 * 3; ch++) {
          sh0_first[ch * n + idx] = 0.f;
          sh0_second[ch * n + idx] = 0.f;
        }
        for (int ch = 0; ch < 3 * 3; ch++) {
          sh1_first[ch * n + idx] = 0.f;
          sh1_second[ch * n + idx] = 0.f;
        }
        for (int ch = 0; ch < 5 * 3; ch++) {
          sh2_first[ch * n + idx] = 0.f;
          sh2_second[ch * n + idx] = 0.f;
        }
        for (int ch = 0; ch < 7 * 3; ch++) {
          sh3_first[ch * n + idx] = 0.f;
          sh3_second[ch * n + idx] = 0.f;
        }
      });
}

void AdamPerGaussian::reset_opacity() {
  thrust::fill(m_opacities_first.begin(), m_opacities_first.end(), 0.f);
  thrust::fill(m_opacities_second.begin(), m_opacities_second.end(), 0.f);
  thrust::fill(m_steps.begin(), m_steps.end(), 0);
}

void AdamPerGaussian::set_params(const json& config) {
  OptimizerBase::set_params(config);
  m_adam_params.from_json(config);
}

json AdamPerGaussian::get_params() const {
  json params = OptimizerBase::get_params();
  params["type"] = "adam_per_gaussian";
  json adam_params = m_adam_params.to_json();
  for (auto& [key, value] : adam_params.items()) {
    params[key] = value;
  }
  return params;
}

void AdamPerGaussian::reorder(uint* indices) {
  int num_gaussians = m_means_first.size();

  thrust::device_vector<vec3> means_first(num_gaussians);
  thrust::device_vector<vec3> means_second(num_gaussians);
  thrust::device_vector<float> opacities_first(num_gaussians);
  thrust::device_vector<float> opacities_second(num_gaussians);
  thrust::device_vector<vec4> rotations_first(num_gaussians);
  thrust::device_vector<vec4> rotations_second(num_gaussians);
  thrust::device_vector<vec3> scales_first(num_gaussians);
  thrust::device_vector<vec3> scales_second(num_gaussians);

  const int grid = (num_gaussians + block_size - 1) / block_size;
  copy_optimizer_base_state_pg<<<grid, block_size>>>(
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
      num_gaussians);

  m_means_first = std::move(means_first);
  m_means_second = std::move(means_second);
  m_opacities_first = std::move(opacities_first);
  m_opacities_second = std::move(opacities_second);
  m_rotations_first = std::move(rotations_first);
  m_rotations_second = std::move(rotations_second);
  m_scales_first = std::move(scales_first);
  m_scales_second = std::move(scales_second);

  thrust::device_vector<uint32_t> new_steps(num_gaussians, 0);
  thrust::device_ptr<uint> indices_ptr(indices);
  thrust::gather(thrust::device, indices_ptr, indices_ptr + num_gaussians, m_steps.begin(), new_steps.begin());
  m_steps = std::move(new_steps);

  thrust::device_vector<float> sh0_f, sh0_s, sh1_f, sh1_s, sh2_f, sh2_s, sh3_f, sh3_s;
  gather_soa_optim_buffers(
      m_sh0_first, m_sh0_second, sh0_f, sh0_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(0) * 3,
      num_gaussians);
  gather_soa_optim_buffers(
      m_sh1_first, m_sh1_second, sh1_f, sh1_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(1) * 3,
      num_gaussians);
  gather_soa_optim_buffers(
      m_sh2_first, m_sh2_second, sh2_f, sh2_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(2) * 3,
      num_gaussians);
  gather_soa_optim_buffers(
      m_sh3_first, m_sh3_second, sh3_f, sh3_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(3) * 3,
      num_gaussians);
  m_sh0_first = std::move(sh0_f);
  m_sh0_second = std::move(sh0_s);
  m_sh1_first = std::move(sh1_f);
  m_sh1_second = std::move(sh1_s);
  m_sh2_first = std::move(sh2_f);
  m_sh2_second = std::move(sh2_s);
  m_sh3_first = std::move(sh3_f);
  m_sh3_second = std::move(sh3_s);
}

}  // namespace tinygs
