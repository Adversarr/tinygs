#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <limits>
#include <memory>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adamw.hpp"

namespace tinygs {

__constant__ float learning_rates[12]; // 3mean 1opa 4rot 3scale 1sh

using vectorize_t = __int128;
constexpr uint32_t per_gaussian_size = sizeof(AosGaussianAdam) / sizeof(vectorize_t);

__device__ __forceinline__ void fast_zero(AosGaussianAdam* gaussian) {
#pragma unroll per_gaussian_size
  for (uint32_t i = 0; i < per_gaussian_size; ++i) {
    ((__int128*)gaussian)[i] = 0;
  }
}

__device__ __forceinline__ void fast_copy(AosGaussianAdam* dst, const AosGaussianAdam* src) {
#pragma unroll per_gaussian_size
  for (uint32_t i = 0; i < per_gaussian_size; ++i) {
    ((__int128*)dst)[i] = ((__int128*)src)[i];
  }
}

template <typename Elem>
__forceinline__ __device__ void adam_step_func(
  Elem& weight,
  Elem gradient,
  Elem& first_moment,
  Elem& second_moment,
  float learning_rate,
  const float& beta1,
  const float& beta2,
  const float& epsilon,
  const float& gradient_clipping_magnitude,
  const float& lower_lr_bound,
  const float& upper_lr_bound,
  const float& this_lr_scale
) {
  if (gradient_clipping_magnitude != 0.0f) {
    gradient = copysign(min(abs(gradient), gradient_clipping_magnitude), gradient);
  }

  const Elem gradient_sq = gradient * gradient;
  first_moment = beta1 * first_moment + (1 - beta1) * gradient;
  second_moment = beta2 * second_moment + (1 - beta2) * gradient_sq;

  // Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
  learning_rate *= this_lr_scale;

  // // Follow AdaBound paradigm
  const Elem effective_learning_rate
      = min(max(learning_rate / (sqrt(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

  weight -= effective_learning_rate * first_moment;
}


// AoS-based Adam step kernel: first moments stored at adam_states[2*idx],
// second moments stored at adam_states[2*idx+1]
__global__ void launch_gaussian_adam_step_AoS(
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
  // Adam state (AoS)
  AosGaussianAdam* __restrict__ adam_states,
  // other
  AdamWParameters adam_p,
  GaussianOptimizationParams general_p,
  uint32_t num_gaussians,
  const float gradient_scale,
  const float global_lr
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians) return;
  const float inv_n = 1.0f / num_gaussians;

  // accumulate all the gradients, to cull out zero gradient gaussians
  const float grad_norm_1 = (
    sum(abs(means_grad[idx]))
    // + abs(opacities_grad[idx])
    // + sum(abs(rotations_grad[idx]))
    // + sum(abs(scales_grad[idx]))
  );
  if (grad_norm_1 == 0 && general_p.skip_zero_grad) return;

  const float beta1 = adam_p.beta1;
  const float beta2 = adam_p.beta2;

  // actually perform the optimization for this gaussian
  AosGaussianAdam first;
  AosGaussianAdam second;
  fast_copy(&first, adam_states+idx * 2);
  const auto this_step = (++first.count);
  const float this_lr_scale = sqrtf(1 - powf(beta2, (float)this_step)) / (1 - powf(beta1, (float)this_step));
  fast_copy(&second, adam_states+idx * 2 + 1);

  // AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
  float lower_lr_bound = 0;
  float upper_lr_bound = std::numeric_limits<float>::max();
  if (adam_p.enable_adabound) {
    lower_lr_bound = 0.1f - 0.1f / ((1 - adam_p.beta2) * (float)this_step + 1);
    upper_lr_bound = 0.1f + 0.1f / ((1 - adam_p.beta2) * (float)this_step);
  }

  { // means
    vec3& val = means[idx];
    const vec3 grad = means_grad[idx] * gradient_scale;
    vec3& first_moment = first.mean;
    vec3& second_moment = second.mean;
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.means_lr * global_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.max_grad_1,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // opacities
    float& val = opacities[idx];
    float actual = activate_opacity(val);
    float grad = opacities_grad[idx] * gradient_scale + (general_p.opacities_l1 * activate_opacity_deriv(val)) * inv_n;
    float& first_moment = first.opacity;
    float& second_moment = second.opacity;
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.opacities_lr * global_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.max_grad_1,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // rotations
    vec4& val = rotations[idx];
    const vec4 grad = rotations_grad[idx] * gradient_scale;
    vec4& first_moment = first.rotation;
    vec4& second_moment = second.rotation;
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.rotations_lr * global_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.max_grad_1,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // scales
    vec3& val = scales[idx];
    vec3 grad = scales_grad[idx] * gradient_scale + (general_p.scales_l1 * vec3(activate_scale_deriv(val.x), activate_scale_deriv(val.y), activate_scale_deriv(val.z))) * inv_n;
    vec3& first_moment = first.scale;
    vec3& second_moment = second.scale;
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.scales_lr * global_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.max_grad_1,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // spherical harmonics - 0th coefficient
    vec3& val = sh_coefficient_0[idx];
    const vec3 grad = sh_coefficient_0_grad[idx] * gradient_scale;
    vec3& first_moment = first.shs[0];
    vec3& second_moment = second.shs[0];
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.shs_lr * global_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.max_grad_1,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // spherical harmonics - rest coefficients
    // NOTE: They use 1/20 LR w.r.t. sh0
    int start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
    int end = start + (kMaxSphericalHarmonicsCoefficients - 1);
    for (int i = start; i < end; i++) {
      const int j = i - start + 1; // SH rest maps to shs[1..]
      vec3& val = sh_coefficients_rest[i];
      const vec3 grad = sh_coefficients_rest_grad[i] * gradient_scale;
      vec3& first_moment = first.shs[j];
      vec3& second_moment = second.shs[j];
      adam_step_func(
        val, grad, first_moment, second_moment,
        general_p.shs_lr * 0.05f * global_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
        general_p.max_grad_1,
        lower_lr_bound,
        upper_lr_bound,
        this_lr_scale
      );
    }
  }

  fast_copy(adam_states+idx * 2, &first);
  fast_copy(adam_states+idx * 2 + 1, &second);
}


void AdamW::step(float scale) {
  const float gradient_scale = scale;  // This is the gradient scaler, not learning rate multiplier
  const int blocks = 64;
  const int grid = (m_gaussians->size() + blocks - 1) / blocks;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("AdamW::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("AdamW::step: gaussians and gaussians_grad must have same size");
  }

  cudaFuncSetCacheConfig(launch_gaussian_adam_step_AoS, cudaFuncCachePreferShared);
  cudaFuncSetAttribute(launch_gaussian_adam_step_AoS, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);

  launch_gaussian_adam_step_AoS<<<grid, blocks>>>(
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
    m_gaussians_adam->data(),
    m_adam_params,
    m_params,
    m_gaussians->size(),
    gradient_scale,
    m_global_lr
  );
}

AdamW::AdamW(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
  OptimizerBase(gaussians, gaussians_grad) {
  // Resize and reset all internal buffers
  AdamW::reset();
}

// Copy AoS Adam states according to mapping (kept list)
__global__ void copy_items_aos(
  const AosGaussianAdam* __restrict__ src,
  AosGaussianAdam* __restrict__ dst,
  const int * __restrict__ mapping,
  int num_kept
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kept) return;
  int src_idx = mapping[idx];
  // Copy first and second moments (interleaved)
  dst[idx * 2] = src[src_idx * 2];
  dst[idx * 2 + 1] = src[src_idx * 2 + 1];
}

void AdamW::remove(char* kept_flag, int num_kept) {
  // filters the gaussians' first second.
  size_t original_size = m_gaussians->size();
  thrust::device_vector<int> mapping(original_size); // mapping[idx] = original_idx

  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(original_size),
    mapping.begin(), [kept_flag] __device__ (int orig) { return static_cast<bool>(kept_flag[orig]); });

  // Allocate new AoS buffer for kept items
  auto new_states = std::make_unique<GPUBuffer<AosGaussianAdam>>(static_cast<size_t>(2 * num_kept));

  const int grid = (num_kept + 255) / 256;
  copy_items_aos<<<grid, 256>>>(
      m_gaussians_adam->data(),
      new_states->data(),
      thrust::raw_pointer_cast(mapping.data()),
      num_kept
  );

  // Swap in new buffer
  m_gaussians_adam = std::move(new_states);
  CUDA_CHECK_THROW(cudaDeviceSynchronize()); CUDA_CHECK_THROW(cudaGetLastError());
}

// Duplicate AoS optimizer state for new gaussians
__global__ static void duplicate_optimizer_state_kernel_aos(
  AosGaussianAdam* __restrict__ states,
  const int* __restrict__ indices,
  const int* __restrict__ new_indices,
  uint32_t num_duplicate
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_duplicate) return;
  const int src_idx = indices[idx];
  const int dst_idx = new_indices[idx];
  constexpr float kHalf = 0.45f;
  constexpr float kQuarter = 0.20f;
  // Copy first and second moments with damping
  AosGaussianAdam first_src = states[src_idx * 2];
  AosGaussianAdam second_src = states[src_idx * 2 + 1];
  // Apply damping
  first_src.mean *= kHalf;
  first_src.opacity *= kHalf;
  first_src.rotation *= kHalf;
  first_src.scale *= kHalf;
  for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients; ++i) first_src.shs[i] *= kHalf;
  second_src.mean *= kQuarter;
  second_src.opacity *= kQuarter;
  second_src.rotation *= kQuarter;
  second_src.scale *= kQuarter;
  for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients; ++i) second_src.shs[i] *= kQuarter;
  // Set
  // states[dst_idx * 2] = first_src;
  // states[dst_idx * 2 + 1] = second_src;
  fast_copy(states + dst_idx * 2, &first_src);
  fast_copy(states + dst_idx * 2 + 1, &second_src);
  // Copy step count (keep same as source)
  states[dst_idx * 2].count = states[src_idx * 2].count;
}

// Copy old AoS states into a larger buffer (preserve existing entries)
__global__ static void preserve_old_states_kernel(
  const AosGaussianAdam* __restrict__ old_states,
  AosGaussianAdam* __restrict__ new_states,
  uint32_t old_num_gaussians
) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= old_num_gaussians) return;
  // // copy first and second moments (interleaved)
  // new_states[i * 2] = old_states[i * 2];
  // new_states[i * 2 + 1] = old_states[i * 2 + 1];
  fast_copy(new_states + i * 2, old_states + i * 2);
  fast_copy(new_states + i * 2 + 1, old_states + i * 2 + 1);
}

void AdamW::duplicate(int* indices, int* new_indices, int num_duplicate) {
  if (num_duplicate == 0) return;
  const uint32_t old_n = static_cast<uint32_t>(m_gaussians_adam ? (m_gaussians_adam->size() / 2) : 0);
  const uint32_t new_n = static_cast<uint32_t>(m_gaussians->size());
  // Allocate new buffer and preserve existing states
  auto new_states = std::make_unique<GPUBuffer<AosGaussianAdam>>(static_cast<size_t>(2 * new_n));
  if (old_n > 0) {
    const int grid_copy = (old_n + 255) / 256;
    preserve_old_states_kernel<<<grid_copy, 256>>>(
      m_gaussians_adam->data(),
      new_states->data(),
      old_n
    );
  }

  // Duplicate selected indices into new slots
  const int grid = (num_duplicate + 255) / 256;
  duplicate_optimizer_state_kernel_aos<<<grid, 256>>>(
    new_states->data(),
    indices,
    new_indices,
    static_cast<uint32_t>(num_duplicate)
  );
  // Swap in new buffer
  m_gaussians_adam = std::move(new_states);
  CUDA_CHECK_THROW(cudaDeviceSynchronize()); CUDA_CHECK_THROW(cudaGetLastError());
}

__global__ void reset_adamw_state_kernel(
  AosGaussianAdam* inout,
  int* indices,
  int num_reset
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_reset) return;
  int gaussian_idx = indices[idx];
  // Zero both first and second moments for this gaussian (interleaved)
  fast_zero(inout + gaussian_idx * 2);
  fast_zero(inout + gaussian_idx * 2 + 1);
}

// __global__ void reset_adamw_state_kernel_all(AosGaussianAdam* inout, size_t num_states) {
//   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx >= num_states) return;
//   // Zero out the entire state at this index
//   fast_zero(inout + idx);
// }

void AdamW::reset() {
  size_t num_gaussians = m_gaussians->size();
  if (!m_gaussians_adam) {
    m_gaussians_adam = std::make_unique<GPUBuffer<AosGaussianAdam>>(static_cast<size_t>(2 * num_gaussians));
  }

  // memset zero
  CUDA_CHECK_THROW(cudaMemset(m_gaussians_adam->data(), 0, sizeof(AosGaussianAdam) * num_gaussians * 2));
}

void AdamW::reset(int* indices, int num_reset) {
  size_t num_gaussians = m_gaussians->size();
  const int grid = (num_reset + 255) / 256;
  reset_adamw_state_kernel<<<grid, 256>>>(
    m_gaussians_adam->data(),
    indices,
    num_reset
  );
}

__global__ void reset_opacity_kernel(AosGaussianAdam* inout, size_t num_gaussians) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians * 2) return;
  inout[idx].opacity = 0.f;
}

void AdamW::reset_opacity() {
  size_t num_gaussians = m_gaussians->size();
  const int grid = (num_gaussians * 2 + 255) / 256;
  reset_opacity_kernel<<<grid, 256>>>(
    m_gaussians_adam->data(),
    num_gaussians
  );
}

void AdamW::set_params(const json& config) {
  // Update base optimizer parameters
  OptimizerBase::set_params(config);
  
  // Update AdamW-specific parameters
  m_adam_params.from_json(config);
}

json AdamW::get_params() const {
  // Get base optimizer parameters
  json params = OptimizerBase::get_params();
  
  // Add type for reflection
  params["type"] = "adam";
  
  // Add AdamW-specific parameters
  json adamw_params = m_adam_params.to_json();
  
  // Merge the two JSON objects
  for (auto& [key, value] : adamw_params.items()) {
    params[key] = value;
  }
  
  return params;
}

json AdamWParameters::to_json() const {
  json j;
  j["beta1"] = beta1;
  j["beta2"] = beta2;
  j["epsilon"] = epsilon;
  j["enable_adabound"] = enable_adabound;
  return j;
}

void AdamWParameters::from_json(const json& config) {
  if (config.contains("beta1")) {
    beta1 = config["beta1"];
  }
  if (config.contains("beta2")) {
    beta2 = config["beta2"];
  }
  if (config.contains("epsilon")) {
    epsilon = config["epsilon"];
  }
  if (config.contains("enable_adabound")) {
    enable_adabound = config["enable_adabound"];
  }
}

} // namespace tinygs