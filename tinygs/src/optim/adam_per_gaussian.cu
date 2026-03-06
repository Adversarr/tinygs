#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>

#include <cmath>
#include <memory>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adam_per_gaussian.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "soa_optim_helpers.cuh"

#include "../helper_math.h"

constexpr int block_size = 512;

namespace tinygs {

struct AdamPerGaussian::Impl {
  std::shared_ptr<BackendBuffer> m_steps;

  std::shared_ptr<BackendBuffer> m_means_first;
  std::shared_ptr<BackendBuffer> m_opacities_first;
  std::shared_ptr<BackendBuffer> m_rotations_first;
  std::shared_ptr<BackendBuffer> m_scales_first;
  std::shared_ptr<BackendBuffer> m_sh0_first;
  std::shared_ptr<BackendBuffer> m_sh1_first;
  std::shared_ptr<BackendBuffer> m_sh2_first;
  std::shared_ptr<BackendBuffer> m_sh3_first;

  std::shared_ptr<BackendBuffer> m_means_second;
  std::shared_ptr<BackendBuffer> m_opacities_second;
  std::shared_ptr<BackendBuffer> m_rotations_second;
  std::shared_ptr<BackendBuffer> m_scales_second;
  std::shared_ptr<BackendBuffer> m_sh0_second;
  std::shared_ptr<BackendBuffer> m_sh1_second;
  std::shared_ptr<BackendBuffer> m_sh2_second;
  std::shared_ptr<BackendBuffer> m_sh3_second;
  
  size_t m_size = 0;
};

#define m_steps m_impl->m_steps
#define m_means_first m_impl->m_means_first
#define m_opacities_first m_impl->m_opacities_first
#define m_rotations_first m_impl->m_rotations_first
#define m_scales_first m_impl->m_scales_first
#define m_sh0_first m_impl->m_sh0_first
#define m_sh1_first m_impl->m_sh1_first
#define m_sh2_first m_impl->m_sh2_first
#define m_sh3_first m_impl->m_sh3_first
#define m_means_second m_impl->m_means_second
#define m_opacities_second m_impl->m_opacities_second
#define m_rotations_second m_impl->m_rotations_second
#define m_scales_second m_impl->m_scales_second
#define m_sh0_second m_impl->m_sh0_second
#define m_sh1_second m_impl->m_sh1_second
#define m_sh2_second m_impl->m_sh2_second
#define m_sh3_second m_impl->m_sh3_second
#define m_size m_impl->m_size

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

void AdamPerGaussian::step(float scale, BackendStream stream) {
  if (m_adam_params.decouple_decay) {
    step_adamw(scale, stream);
  } else {
    step_adam(scale, stream);
  }
}

void AdamPerGaussian::step(const GroupStepConfig& step_config, BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
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

  increment_steps<<<div_round_up<uint>(n, launch_block_size), launch_block_size, 0, cuda_stream>>>(
      buffer_data<uint32_t>(m_steps), n);

  const auto* steps_ptr = buffer_data<uint32_t>(m_steps);
  const bool use_adamw = m_adam_params.decouple_decay;

  if (step_config.update_means) {
    if (use_adamw) {
      adamw_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          reinterpret_cast<float*>(m_gaussians->means().data()),
          reinterpret_cast<const float*>(m_gaussians_grad->means().data()),
          buffer_data<float>(m_means_first),
          buffer_data<float>(m_means_second),
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
      adam_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          reinterpret_cast<float*>(m_gaussians->means().data()),
          reinterpret_cast<const float*>(m_gaussians_grad->means().data()),
          buffer_data<float>(m_means_first),
          buffer_data<float>(m_means_second),
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
      adamw_pg<<<div_round_up<uint>(n, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->opacities().data(),
          m_gaussians_grad->opacities().data(),
          buffer_data<float>(m_opacities_first),
          buffer_data<float>(m_opacities_second),
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
      adam_pg<<<div_round_up<uint>(n, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->opacities().data(),
          m_gaussians_grad->opacities().data(),
          buffer_data<float>(m_opacities_first),
          buffer_data<float>(m_opacities_second),
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
      adamw_pg<<<div_round_up<uint>(n * 4, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          reinterpret_cast<float*>(m_gaussians->rotations().data()),
          reinterpret_cast<const float*>(m_gaussians_grad->rotations().data()),
          buffer_data<float>(m_rotations_first),
          buffer_data<float>(m_rotations_second),
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
      adam_pg<<<div_round_up<uint>(n * 4, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          reinterpret_cast<float*>(m_gaussians->rotations().data()),
          reinterpret_cast<const float*>(m_gaussians_grad->rotations().data()),
          buffer_data<float>(m_rotations_first),
          buffer_data<float>(m_rotations_second),
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
      adamw_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          reinterpret_cast<float*>(m_gaussians->scales().data()),
          reinterpret_cast<const float*>(m_gaussians_grad->scales().data()),
          buffer_data<float>(m_scales_first),
          buffer_data<float>(m_scales_second),
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
      adam_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          reinterpret_cast<float*>(m_gaussians->scales().data()),
          reinterpret_cast<const float*>(m_gaussians_grad->scales().data()),
          buffer_data<float>(m_scales_first),
          buffer_data<float>(m_scales_second),
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
      adamw_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh0().data(),
          m_gaussians_grad->sh0().data(),
          buffer_data<float>(m_sh0_first),
          buffer_data<float>(m_sh0_second),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_shs_global_lr,
          n * 3,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1);
      adamw_pg<<<div_round_up<uint>(n * 9, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh1().data(),
          m_gaussians_grad->sh1().data(),
          buffer_data<float>(m_sh1_first),
          buffer_data<float>(m_sh1_second),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh1_lr_scale * m_shs_global_lr,
          n * 9,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1);
      adamw_pg<<<div_round_up<uint>(n * 15, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh2().data(),
          m_gaussians_grad->sh2().data(),
          buffer_data<float>(m_sh2_first),
          buffer_data<float>(m_sh2_second),
          steps_ptr,
          m_adam_params,
          m_params.shs_lr * m_params.sh2_lr_scale * m_shs_global_lr,
          n * 15,
          n,
          1,
          kGaussianIndexMod,
          step_config.shs_scale,
          m_params.max_grad_1);
      adamw_pg<<<div_round_up<uint>(n * 21, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh3().data(),
          m_gaussians_grad->sh3().data(),
          buffer_data<float>(m_sh3_first),
          buffer_data<float>(m_sh3_second),
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
      adam_pg<<<div_round_up<uint>(n * 3, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh0().data(),
          m_gaussians_grad->sh0().data(),
          buffer_data<float>(m_sh0_first),
          buffer_data<float>(m_sh0_second),
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
      adam_pg<<<div_round_up<uint>(n * 9, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh1().data(),
          m_gaussians_grad->sh1().data(),
          buffer_data<float>(m_sh1_first),
          buffer_data<float>(m_sh1_second),
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
      adam_pg<<<div_round_up<uint>(n * 15, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh2().data(),
          m_gaussians_grad->sh2().data(),
          buffer_data<float>(m_sh2_first),
          buffer_data<float>(m_sh2_second),
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
      adam_pg<<<div_round_up<uint>(n * 21, launch_block_size), launch_block_size, 0, cuda_stream>>>(
          m_gaussians->sh3().data(),
          m_gaussians_grad->sh3().data(),
          buffer_data<float>(m_sh3_first),
          buffer_data<float>(m_sh3_second),
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

void AdamPerGaussian::step_adam(float scale, BackendStream stream) {
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

void AdamPerGaussian::step_adamw(float scale, BackendStream stream) {
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

AdamPerGaussian::AdamPerGaussian(std::shared_ptr<BackendRuntime> runtime,
                                 std::shared_ptr<GPUGaussian3d> gaussians,
                                 std::shared_ptr<GPUGaussian3d> gaussians_grad)
    : OptimizerBase(std::move(runtime), gaussians, gaussians_grad),
      m_impl(std::make_unique<Impl>()) {
  QueueDesc queue_desc;
  queue_desc.debug_name = "adam_pg_init_queue";
  auto queue_result = m_runtime->create_queue(queue_desc);
  if (!queue_result.ok()) {
    throw std::runtime_error("AdamPerGaussian: failed to create init queue: " + to_string(queue_result.error()));
  }
  AdamPerGaussian::reset(queue_result.value());
  m_runtime->synchronize_queue(queue_result.value());
}

AdamPerGaussian::~AdamPerGaussian() = default;

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

void AdamPerGaussian::remove(char* kept_flag, int num_kept, const std::shared_ptr<BackendQueue>& queue) {
  size_t original_size = m_size;
  auto mapping = create_device_buffer_for<uint>(m_runtime, original_size, "remove_mapping");

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(queue->native_handle());
  auto end_it = thrust::copy_if(thrust::cuda::par.on(stream),
                                thrust::make_counting_iterator<uint>(0),
                                thrust::make_counting_iterator<uint>(original_size),
                                buffer_data<uint>(mapping),
                                [kept_flag] __device__(uint orig) { return static_cast<bool>(kept_flag[orig]); });
  const int kept_from_mapping = static_cast<int>(end_it - buffer_data<uint>(mapping));
  if (kept_from_mapping != num_kept) {
    throw std::runtime_error("AdamPerGaussian::remove: num_kept mismatch with keep mask");
  }

  auto new_means_first = create_device_buffer_for<vec3>(m_runtime, num_kept, "means_first");
  auto new_means_second = create_device_buffer_for<vec3>(m_runtime, num_kept, "means_second");
  auto new_opacities_first = create_device_buffer_for<float>(m_runtime, num_kept, "opacities_first");
  auto new_opacities_second = create_device_buffer_for<float>(m_runtime, num_kept, "opacities_second");
  auto new_rotations_first = create_device_buffer_for<vec4>(m_runtime, num_kept, "rotations_first");
  auto new_rotations_second = create_device_buffer_for<vec4>(m_runtime, num_kept, "rotations_second");
  auto new_scales_first = create_device_buffer_for<vec3>(m_runtime, num_kept, "scales_first");
  auto new_scales_second = create_device_buffer_for<vec3>(m_runtime, num_kept, "scales_second");

  const int grid = (num_kept + block_size - 1) / block_size;
  copy_optimizer_base_state_pg<<<grid, block_size, 0, stream>>>(
      buffer_data<vec3>(m_means_first),
      buffer_data<vec3>(m_means_second),
      buffer_data<vec3>(new_means_first),
      buffer_data<vec3>(new_means_second),
      buffer_data<float>(m_opacities_first),
      buffer_data<float>(m_opacities_second),
      buffer_data<float>(new_opacities_first),
      buffer_data<float>(new_opacities_second),
      buffer_data<vec4>(m_rotations_first),
      buffer_data<vec4>(m_rotations_second),
      buffer_data<vec4>(new_rotations_first),
      buffer_data<vec4>(new_rotations_second),
      buffer_data<vec3>(m_scales_first),
      buffer_data<vec3>(m_scales_second),
      buffer_data<vec3>(new_scales_first),
      buffer_data<vec3>(new_scales_second),
      buffer_data<uint>(mapping),
      num_kept);

  m_means_first = std::move(new_means_first);
  m_means_second = std::move(new_means_second);
  m_opacities_first = std::move(new_opacities_first);
  m_opacities_second = std::move(new_opacities_second);
  m_rotations_first = std::move(new_rotations_first);
  m_rotations_second = std::move(new_rotations_second);
  m_scales_first = std::move(new_scales_first);
  m_scales_second = std::move(new_scales_second);

  auto new_steps = create_device_buffer_for<uint32_t>(m_runtime, num_kept, "steps");
  fill_buffer_zero_async(m_runtime, queue, new_steps);
  thrust::gather(thrust::cuda::par.on(stream),
                 buffer_data<uint>(mapping),
                 buffer_data<uint>(mapping) + num_kept,
                 buffer_data<uint32_t>(m_steps),
                 buffer_data<uint32_t>(new_steps));
  m_steps = std::move(new_steps);

  const uint* map_ptr = buffer_data<uint>(mapping);
  const int old_n = static_cast<int>(original_size);
  std::shared_ptr<BackendBuffer> sh0_f, sh0_s, sh1_f, sh1_s, sh2_f, sh2_s, sh3_f, sh3_s;
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh0_first, m_sh0_second, sh0_f, sh0_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(0) * 3, old_n,
      block_size);
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh1_first, m_sh1_second, sh1_f, sh1_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(1) * 3, old_n,
      block_size);
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh2_first, m_sh2_second, sh2_f, sh2_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(2) * 3, old_n,
      block_size);
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh3_first, m_sh3_second, sh3_f, sh3_s, map_ptr, num_kept, GPUGaussian3d::sh_degree_num_coeffs(3) * 3, old_n,
      block_size);
  m_sh0_first = std::move(sh0_f);
  m_sh0_second = std::move(sh0_s);
  m_sh1_first = std::move(sh1_f);
  m_sh1_second = std::move(sh1_s);
  m_sh2_first = std::move(sh2_f);
  m_sh2_second = std::move(sh2_s);
  m_sh3_first = std::move(sh3_f);
  m_sh3_second = std::move(sh3_s);
  
  m_size = num_kept;
}

void AdamPerGaussian::duplicate(int* indices, int* new_indices, int num_duplicate, const std::shared_ptr<BackendQueue>& queue) {
  if (num_duplicate == 0) return;
  CHECK_THROW(indices != nullptr);
  CHECK_THROW(new_indices != nullptr);

  const int old_n = static_cast<int>(m_size);
  const int new_n = static_cast<int>(m_gaussians->size());

  m_means_first = resize_buffer_async<vec3>(m_runtime, queue, m_means_first, new_n, "means_first");
  m_means_second = resize_buffer_async<vec3>(m_runtime, queue, m_means_second, new_n, "means_second");
  m_opacities_first = resize_buffer_async<float>(m_runtime, queue, m_opacities_first, new_n, "opacities_first");
  m_opacities_second = resize_buffer_async<float>(m_runtime, queue, m_opacities_second, new_n, "opacities_second");
  m_rotations_first = resize_buffer_async<vec4>(m_runtime, queue, m_rotations_first, new_n, "rotations_first");
  m_rotations_second = resize_buffer_async<vec4>(m_runtime, queue, m_rotations_second, new_n, "rotations_second");
  m_scales_first = resize_buffer_async<vec3>(m_runtime, queue, m_scales_first, new_n, "scales_first");
  m_scales_second = resize_buffer_async<vec3>(m_runtime, queue, m_scales_second, new_n, "scales_second");
  m_steps = resize_buffer_async<uint32_t>(m_runtime, queue, m_steps, new_n, "steps");

  for (int deg = 0; deg < 4; deg++) {
    int nc = GPUGaussian3d::sh_degree_num_coeffs(deg) * 3;
    auto relayout_pair = [&](std::shared_ptr<BackendBuffer>& first, std::shared_ptr<BackendBuffer>& second) {
      optim_detail::relayout_soa_optim(m_runtime, queue, first, old_n, new_n, nc, block_size);
      optim_detail::relayout_soa_optim(m_runtime, queue, second, old_n, new_n, nc, block_size);
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

  if (m_adam_params.copy_state_on_duplicate) {
    optim_detail::duplicate_optim_buffer<uint32_t>(queue, m_steps, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<vec3>(queue, m_means_first, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<vec3>(queue, m_means_second, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<float>(queue, m_opacities_first, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<float>(queue, m_opacities_second, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<vec4>(queue, m_rotations_first, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<vec4>(queue, m_rotations_second, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<vec3>(queue, m_scales_first, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_optim_buffer<vec3>(queue, m_scales_second, indices, new_indices, num_duplicate, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh0_first, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(0) * 3, new_n, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh0_second, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(0) * 3, new_n, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh1_first, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(1) * 3, new_n, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh1_second, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(1) * 3, new_n, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh2_first, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(2) * 3, new_n, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh2_second, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(2) * 3, new_n, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh3_first, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(3) * 3, new_n, block_size);
    optim_detail::duplicate_soa_optim_buffer(
        queue, m_sh3_second, indices, new_indices, num_duplicate, GPUGaussian3d::sh_degree_num_coeffs(3) * 3, new_n, block_size);
  }
  
  m_size = new_n;
}

void AdamPerGaussian::reset(const std::shared_ptr<BackendQueue>& queue) {
  size_t num_gaussians = m_gaussians->size();
  m_size = num_gaussians;

  m_steps = create_device_buffer_for<uint32_t>(m_runtime, num_gaussians, "steps");
  m_means_first = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "means_first");
  m_means_second = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "means_second");
  m_opacities_first = create_device_buffer_for<float>(m_runtime, num_gaussians, "opacities_first");
  m_opacities_second = create_device_buffer_for<float>(m_runtime, num_gaussians, "opacities_second");
  m_rotations_first = create_device_buffer_for<vec4>(m_runtime, num_gaussians, "rotations_first");
  m_rotations_second = create_device_buffer_for<vec4>(m_runtime, num_gaussians, "rotations_second");
  m_scales_first = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "scales_first");
  m_scales_second = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "scales_second");
  m_sh0_first = create_device_buffer_for<float>(m_runtime, num_gaussians * 3, "sh0_first");
  m_sh0_second = create_device_buffer_for<float>(m_runtime, num_gaussians * 3, "sh0_second");
  m_sh1_first = create_device_buffer_for<float>(m_runtime, num_gaussians * 9, "sh1_first");
  m_sh1_second = create_device_buffer_for<float>(m_runtime, num_gaussians * 9, "sh1_second");
  m_sh2_first = create_device_buffer_for<float>(m_runtime, num_gaussians * 15, "sh2_first");
  m_sh2_second = create_device_buffer_for<float>(m_runtime, num_gaussians * 15, "sh2_second");
  m_sh3_first = create_device_buffer_for<float>(m_runtime, num_gaussians * 21, "sh3_first");
  m_sh3_second = create_device_buffer_for<float>(m_runtime, num_gaussians * 21, "sh3_second");

  fill_buffer_zero_async(m_runtime, queue, m_steps);
  fill_buffer_zero_async(m_runtime, queue, m_means_first);
  fill_buffer_zero_async(m_runtime, queue, m_means_second);
  fill_buffer_zero_async(m_runtime, queue, m_opacities_first);
  fill_buffer_zero_async(m_runtime, queue, m_opacities_second);
  fill_buffer_zero_async(m_runtime, queue, m_rotations_first);
  fill_buffer_zero_async(m_runtime, queue, m_rotations_second);
  fill_buffer_zero_async(m_runtime, queue, m_scales_first);
  fill_buffer_zero_async(m_runtime, queue, m_scales_second);
  fill_buffer_zero_async(m_runtime, queue, m_sh0_first);
  fill_buffer_zero_async(m_runtime, queue, m_sh0_second);
  fill_buffer_zero_async(m_runtime, queue, m_sh1_first);
  fill_buffer_zero_async(m_runtime, queue, m_sh1_second);
  fill_buffer_zero_async(m_runtime, queue, m_sh2_first);
  fill_buffer_zero_async(m_runtime, queue, m_sh2_second);
  fill_buffer_zero_async(m_runtime, queue, m_sh3_first);
  fill_buffer_zero_async(m_runtime, queue, m_sh3_second);
}

void AdamPerGaussian::reset(int* indices, int num_reset) {
  const int n = static_cast<int>(m_size);
  thrust::for_each(
      thrust::device_ptr<int>(indices),
      thrust::device_ptr<int>(indices) + num_reset,
      [means_first = buffer_data<vec3>(m_means_first),
       means_second = buffer_data<vec3>(m_means_second),
       opacities_first = buffer_data<float>(m_opacities_first),
       opacities_second = buffer_data<float>(m_opacities_second),
       rotations_first = buffer_data<vec4>(m_rotations_first),
       rotations_second = buffer_data<vec4>(m_rotations_second),
       scales_first = buffer_data<vec3>(m_scales_first),
       scales_second = buffer_data<vec3>(m_scales_second),
       sh0_first = buffer_data<float>(m_sh0_first),
       sh0_second = buffer_data<float>(m_sh0_second),
       sh1_first = buffer_data<float>(m_sh1_first),
       sh1_second = buffer_data<float>(m_sh1_second),
       sh2_first = buffer_data<float>(m_sh2_first),
       sh2_second = buffer_data<float>(m_sh2_second),
       sh3_first = buffer_data<float>(m_sh3_first),
       sh3_second = buffer_data<float>(m_sh3_second),
       steps = buffer_data<uint32_t>(m_steps),
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

void AdamPerGaussian::reset_opacity(const std::shared_ptr<BackendQueue>& queue) {
  fill_buffer_zero_async(m_runtime, queue, m_opacities_first);
  fill_buffer_zero_async(m_runtime, queue, m_opacities_second);
  fill_buffer_zero_async(m_runtime, queue, m_steps);
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

void AdamPerGaussian::reorder(uint* indices, const std::shared_ptr<BackendQueue>& queue) {
  int num_gaussians = static_cast<int>(m_size);

  auto new_means_first = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "means_first");
  auto new_means_second = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "means_second");
  auto new_opacities_first = create_device_buffer_for<float>(m_runtime, num_gaussians, "opacities_first");
  auto new_opacities_second = create_device_buffer_for<float>(m_runtime, num_gaussians, "opacities_second");
  auto new_rotations_first = create_device_buffer_for<vec4>(m_runtime, num_gaussians, "rotations_first");
  auto new_rotations_second = create_device_buffer_for<vec4>(m_runtime, num_gaussians, "rotations_second");
  auto new_scales_first = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "scales_first");
  auto new_scales_second = create_device_buffer_for<vec3>(m_runtime, num_gaussians, "scales_second");

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(queue->native_handle());
  const int grid = (num_gaussians + block_size - 1) / block_size;
  copy_optimizer_base_state_pg<<<grid, block_size, 0, stream>>>(
      buffer_data<vec3>(m_means_first),
      buffer_data<vec3>(m_means_second),
      buffer_data<vec3>(new_means_first),
      buffer_data<vec3>(new_means_second),
      buffer_data<float>(m_opacities_first),
      buffer_data<float>(m_opacities_second),
      buffer_data<float>(new_opacities_first),
      buffer_data<float>(new_opacities_second),
      buffer_data<vec4>(m_rotations_first),
      buffer_data<vec4>(m_rotations_second),
      buffer_data<vec4>(new_rotations_first),
      buffer_data<vec4>(new_rotations_second),
      buffer_data<vec3>(m_scales_first),
      buffer_data<vec3>(m_scales_second),
      buffer_data<vec3>(new_scales_first),
      buffer_data<vec3>(new_scales_second),
      indices,
      num_gaussians);

  m_means_first = std::move(new_means_first);
  m_means_second = std::move(new_means_second);
  m_opacities_first = std::move(new_opacities_first);
  m_opacities_second = std::move(new_opacities_second);
  m_rotations_first = std::move(new_rotations_first);
  m_rotations_second = std::move(new_rotations_second);
  m_scales_first = std::move(new_scales_first);
  m_scales_second = std::move(new_scales_second);

  auto new_steps = create_device_buffer_for<uint32_t>(m_runtime, num_gaussians, "steps");
  fill_buffer_zero_async(m_runtime, queue, new_steps);
  thrust::device_ptr<uint> indices_ptr(indices);
  thrust::gather(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + num_gaussians, 
                 buffer_data<uint32_t>(m_steps), buffer_data<uint32_t>(new_steps));
  m_steps = std::move(new_steps);

  std::shared_ptr<BackendBuffer> sh0_f, sh0_s, sh1_f, sh1_s, sh2_f, sh2_s, sh3_f, sh3_s;
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh0_first, m_sh0_second, sh0_f, sh0_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(0) * 3,
      num_gaussians, block_size);
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh1_first, m_sh1_second, sh1_f, sh1_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(1) * 3,
      num_gaussians, block_size);
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh2_first, m_sh2_second, sh2_f, sh2_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(2) * 3,
      num_gaussians, block_size);
  optim_detail::gather_soa_optim_buffers(
      m_runtime, queue, m_sh3_first, m_sh3_second, sh3_f, sh3_s, indices, num_gaussians, GPUGaussian3d::sh_degree_num_coeffs(3) * 3,
      num_gaussians, block_size);
  m_sh0_first = std::move(sh0_f);
  m_sh0_second = std::move(sh0_s);
  m_sh1_first = std::move(sh1_f);
  m_sh1_second = std::move(sh1_s);
  m_sh2_first = std::move(sh2_f);
  m_sh2_second = std::move(sh2_s);
  m_sh3_first = std::move(sh3_f);
  m_sh3_second = std::move(sh3_s);
}

#undef m_steps
#undef m_means_first
#undef m_opacities_first
#undef m_rotations_first
#undef m_scales_first
#undef m_sh0_first
#undef m_sh1_first
#undef m_sh2_first
#undef m_sh3_first
#undef m_means_second
#undef m_opacities_second
#undef m_rotations_second
#undef m_scales_second
#undef m_sh0_second
#undef m_sh1_second
#undef m_sh2_second
#undef m_sh3_second
#undef m_size

}  // namespace tinygs
