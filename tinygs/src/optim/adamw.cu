#include <thrust/execution_policy.h>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adamw.hpp"


namespace tinygs {

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




__global__ void launch_gaussian_adam_step_SoA(
  // Means
  vec3* __restrict__ means,
  const vec3* __restrict__ means_grad,
  vec3* __restrict__ means_first_second,
  // Opacities
  float* __restrict__ opacities,
  const float* __restrict__ opacities_grad,
  float* __restrict__ opacities_first_second,
  // Rotation
  vec4* __restrict__ rotations,
  const vec4* __restrict__ rotations_grad,
  vec4* __restrict__ rotations_first_second,
  // Scales
  vec3* __restrict__ scales,
  const vec3* __restrict__ scales_grad,
  vec3* __restrict__ scales_first_second,
  // Spherical Harmonics
  vec3* __restrict__ sh_coefficient_0,
  const vec3* __restrict__ sh_coefficient_0_grad,
  vec3* __restrict__ sh_coefficient_0_first_second,
  vec3* __restrict__ sh_coefficients_rest,
  const vec3* __restrict__ sh_coefficients_rest_grad,
  vec3* __restrict__ sh_coefficients_rest_first_second,
  // other
  uint32_t* __restrict__ gaussian_steps,
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
    sum(abs(means_grad[idx])) +
    abs(opacities_grad[idx]) +
    sum(abs(rotations_grad[idx])) +
    sum(abs(scales_grad[idx]))
  );
  if (grad_norm_1 == 0 && general_p.skip_zero_grad) return;

  const float beta1 = adam_p.beta1;
  const float beta2 = adam_p.beta2;

  // actually perform the optimization for this gaussian
  const auto this_step = (++gaussian_steps[idx]);
  const float this_lr_scale = sqrtf(1 - powf(beta2, (float)this_step)) / (1 - powf(beta1, (float)this_step));

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
    vec3& first_moment = means_first_second[idx * 2];
    vec3& second_moment = means_first_second[idx * 2 + 1];
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
    float& first_moment = opacities_first_second[idx * 2];
    float& second_moment = opacities_first_second[idx * 2 + 1];
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
    vec4& first_moment = rotations_first_second[idx * 2];
    vec4& second_moment = rotations_first_second[idx * 2 + 1];
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
    vec3& first_moment = scales_first_second[idx * 2];
    vec3& second_moment = scales_first_second[idx * 2 + 1];
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
    vec3& first_moment = sh_coefficient_0_first_second[idx * 2];
    vec3& second_moment = sh_coefficient_0_first_second[idx * 2 + 1];
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
      vec3& val = sh_coefficients_rest[i];
      const vec3 grad = sh_coefficients_rest_grad[i] * gradient_scale;
      vec3& first_moment = sh_coefficients_rest_first_second[i * 2];
      vec3& second_moment = sh_coefficients_rest_first_second[i * 2 + 1];
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
}

void AdamW::step(float scale) {
  const float gradient_scale = scale;  // This is the gradient scaler, not learning rate multiplier
  const int grid = (m_gaussians->size() + 255) / 256;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("AdamW::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("AdamW::step: gaussians and gaussians_grad must have same size");
  }

  launch_gaussian_adam_step_SoA<<<grid, 256>>>(
    thrust::raw_pointer_cast(m_gaussians->means().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
    thrust::raw_pointer_cast(m_means_first_second.data()),
    thrust::raw_pointer_cast(m_gaussians->opacities().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
    thrust::raw_pointer_cast(m_opacities_first_second.data()),
    thrust::raw_pointer_cast(m_gaussians->rotations().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
    thrust::raw_pointer_cast(m_rotations_first_second.data()),
    thrust::raw_pointer_cast(m_gaussians->scales().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
    thrust::raw_pointer_cast(m_scales_first_second.data()),
    thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
    thrust::raw_pointer_cast(m_sh_coefficient_0_first_second.data()),
    thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
    thrust::raw_pointer_cast(m_sh_coefficients_rest_first_second.data()),
    thrust::raw_pointer_cast(m_gaussian_steps.data()),
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

__global__ void copy_items(
  const vec3 * __restrict__ src_means,
  vec3 * __restrict__ dst_means,
  const float * __restrict__ src_opacities,
  float * __restrict__ dst_opacities,
  const vec4 * __restrict__ src_rotations,
  vec4 * __restrict__ dst_rotations,
  const vec3 * __restrict__ src_scales,
  vec3 * __restrict__ dst_scales,
  const vec3 * __restrict__ src_sh_coefficient_0,
  vec3 * __restrict__ dst_sh_coefficient_0,
  const vec3 * __restrict__ src_sh_coefficients_rest,
  vec3 * __restrict__ dst_sh_coefficients_rest,
  const uint32_t* __restrict__ src_gaussian_steps,
  uint32_t* __restrict__ dst_gaussian_steps,
  const int * __restrict__ mapping,
  int num_kept
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kept) return;

  int src_idx = mapping[idx];
  
  // Copy first/second moments for means
  dst_means[idx * 2] = src_means[src_idx * 2];
  dst_means[idx * 2 + 1] = src_means[src_idx * 2 + 1];

  // Copy first/second moments for opacities  
  dst_opacities[idx * 2] = src_opacities[src_idx * 2];
  dst_opacities[idx * 2 + 1] = src_opacities[src_idx * 2 + 1];

  // Copy first/second moments for rotations
  dst_rotations[idx * 2] = src_rotations[src_idx * 2];
  dst_rotations[idx * 2 + 1] = src_rotations[src_idx * 2 + 1];

  // Copy first/second moments for scales
  dst_scales[idx * 2] = src_scales[src_idx * 2];
  dst_scales[idx * 2 + 1] = src_scales[src_idx * 2 + 1];

  // Copy first/second moments for SH coefficient 0
  dst_sh_coefficient_0[idx * 2] = src_sh_coefficient_0[src_idx * 2];
  dst_sh_coefficient_0[idx * 2 + 1] = src_sh_coefficient_0[src_idx * 2 + 1];

  // Copy gaussian steps
  dst_gaussian_steps[idx] = src_gaussian_steps[src_idx];

  // Copy first/second moments for rest SH coefficients
  int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1) * 2;
  int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1) * 2;
  for (int i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
    dst_sh_coefficients_rest[dst_rest_start + i * 2] = src_sh_coefficients_rest[src_rest_start + i * 2];
    dst_sh_coefficients_rest[dst_rest_start + i * 2 + 1] = src_sh_coefficients_rest[src_rest_start + i * 2 + 1];
  }
}

void AdamW::remove(char* kept_flag, int num_kept) {
  // filters the gaussians' first second.
  size_t original_size = m_gaussians->size();
  thrust::device_vector<int> mapping(original_size); // mapping[idx] = original_idx

  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(original_size),
    mapping.begin(), [kept_flag] __device__ (int orig) { return static_cast<bool>(kept_flag[orig]); });

  thrust::device_vector<vec3> means(2 * num_kept);
  thrust::device_vector<float> opacities(2 * num_kept);
  thrust::device_vector<vec4> rotations(2 * num_kept);
  thrust::device_vector<vec3> scales(2 * num_kept);
  thrust::device_vector<vec3> sh_coefficients_0(2 * num_kept);
  thrust::device_vector<vec3> sh_coefficients_rest(2 * num_kept * (kMaxSphericalHarmonicsCoefficients - 1));
  thrust::device_vector<uint32_t> gaussian_steps(num_kept);

  const int grid = (num_kept + 255) / 256;
  copy_items<<<grid, 256>>>(
      thrust::raw_pointer_cast(m_means_first_second.data()),
      thrust::raw_pointer_cast(means.data()),
      thrust::raw_pointer_cast(m_opacities_first_second.data()),
      thrust::raw_pointer_cast(opacities.data()),
      thrust::raw_pointer_cast(m_rotations_first_second.data()),
      thrust::raw_pointer_cast(rotations.data()),
      thrust::raw_pointer_cast(m_scales_first_second.data()),
      thrust::raw_pointer_cast(scales.data()),
      thrust::raw_pointer_cast(m_sh_coefficient_0_first_second.data()),
      thrust::raw_pointer_cast(sh_coefficients_0.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_first_second.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest.data()),
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      thrust::raw_pointer_cast(gaussian_steps.data()),
      thrust::raw_pointer_cast(mapping.data()),
      num_kept
  );

  m_means_first_second = std::move(means);
  m_opacities_first_second = std::move(opacities);
  m_rotations_first_second = std::move(rotations);
  m_scales_first_second = std::move(scales);
  m_sh_coefficient_0_first_second = std::move(sh_coefficients_0);
  m_gaussian_steps = std::move(gaussian_steps);
  m_sh_coefficients_rest_first_second = std::move(sh_coefficients_rest);
  CUDA_CHECK_THROW(cudaDeviceSynchronize()); CUDA_CHECK_THROW(cudaGetLastError());
}

__global__ static void duplicate_optimizer_state_kernel(
  // Means
  vec3* __restrict__ means_first_second,
  // Opacities
  float* __restrict__ opacities_first_second,
  // Rotations
  vec4* __restrict__ rotations_first_second,
  // Scales
  vec3* __restrict__ scales_first_second,
  // Spherical Harmonics
  vec3* __restrict__ sh_coefficient_0_first_second,
  vec3* __restrict__ sh_coefficients_rest_first_second,
  // Step counts
  uint32_t* __restrict__ gaussian_steps,
  // Indices
  const int* __restrict__ indices,
  const int* __restrict__ new_indices,
  uint32_t num_duplicate,
  uint32_t num_sh_rest_per_gaussian
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_duplicate) return;

  const int src_idx = indices[idx];
  const int dst_idx = new_indices[idx];

  constexpr float kHalf = 0.45f;
  constexpr float kQuarter = 0.20f;

  // Copy means first and second moments
  means_first_second[dst_idx * 2] = means_first_second[src_idx * 2] *= kHalf;
  means_first_second[dst_idx * 2 + 1] = means_first_second[src_idx * 2 + 1] *= kQuarter;

  // Copy opacities first and second moments
  opacities_first_second[dst_idx * 2] = opacities_first_second[src_idx * 2] *= kHalf;
  opacities_first_second[dst_idx * 2 + 1] = opacities_first_second[src_idx * 2 + 1] *= kQuarter;

  // Copy rotations first and second moments
  rotations_first_second[dst_idx * 2] = rotations_first_second[src_idx * 2] *= kHalf;
  rotations_first_second[dst_idx * 2 + 1] = rotations_first_second[src_idx * 2 + 1] *= kQuarter;

  // Copy scales first and second moments
  scales_first_second[dst_idx * 2] = scales_first_second[src_idx * 2] *= kHalf;
  scales_first_second[dst_idx * 2 + 1] = scales_first_second[src_idx * 2 + 1] *= kQuarter;

  // Copy sh_coefficient_0 first and second moments
  sh_coefficient_0_first_second[dst_idx * 2] = sh_coefficient_0_first_second[src_idx * 2] *= kHalf;
  sh_coefficient_0_first_second[dst_idx * 2 + 1] = sh_coefficient_0_first_second[src_idx * 2 + 1] *= kQuarter;

  // Copy sh_coefficients_rest first and second moments
  for (uint32_t i = 0; i < num_sh_rest_per_gaussian; i++) {
    const int src_sh_idx = src_idx * num_sh_rest_per_gaussian + i;
    const int dst_sh_idx = dst_idx * num_sh_rest_per_gaussian + i;
    sh_coefficients_rest_first_second[dst_sh_idx * 2] = sh_coefficients_rest_first_second[src_sh_idx * 2] *= kHalf;
    sh_coefficients_rest_first_second[dst_sh_idx * 2 + 1] = sh_coefficients_rest_first_second[src_sh_idx * 2 + 1] *= kQuarter;
  }

  // Copy step count: but with half the value to ensure it could be optimized efficiently.
  gaussian_steps[dst_idx] = (gaussian_steps[src_idx]);
}

void AdamW::duplicate(int* indices, int* new_indices, int num_duplicate) {
  if (num_duplicate == 0) return;
  const uint32_t num_sh_rest_per_gaussian = kMaxSphericalHarmonicsCoefficients - 1;
  // Resize vectors to accommodate duplicated gaussians
  m_means_first_second.resize(m_gaussians->size() * 2, vec3(0.f, 0.f, 0.f));
  m_opacities_first_second.resize(m_gaussians->size() * 2, 0.f);
  m_rotations_first_second.resize(m_gaussians->size() * 2, vec4(0.f, 0.f, 0.f, 0.f));
  m_scales_first_second.resize(m_gaussians->size() * 2, vec3(0.f, 0.f, 0.f));
  m_sh_coefficient_0_first_second.resize(m_gaussians->size() * 2, vec3(0.f, 0.f, 0.f));
  m_sh_coefficients_rest_first_second.resize(m_gaussians->size() * 2 * num_sh_rest_per_gaussian, vec3(0.f, 0.f, 0.f));
  m_gaussian_steps.resize(m_gaussians->size(), 0);

  // TODO: This design does not provide better result. Why?
  // const int grid = (num_duplicate + 255) / 256;
  // duplicate_optimizer_state_kernel<<<grid, 256>>>(
  //   thrust::raw_pointer_cast(m_means_first_second.data()),
  //   thrust::raw_pointer_cast(m_opacities_first_second.data()),
  //   thrust::raw_pointer_cast(m_rotations_first_second.data()),
  //   thrust::raw_pointer_cast(m_scales_first_second.data()),
  //   thrust::raw_pointer_cast(m_sh_coefficient_0_first_second.data()),
  //   thrust::raw_pointer_cast(m_sh_coefficients_rest_first_second.data()),
  //   thrust::raw_pointer_cast(m_gaussian_steps.data()),
  //   indices,
  //   new_indices,
  //   num_duplicate,
  //   num_sh_rest_per_gaussian
  // );
}

void AdamW::reset() {
  size_t num_gaussians = m_gaussians->size();

  m_means_first_second.resize(num_gaussians * 2, vec3(0.f));
  m_opacities_first_second.resize(num_gaussians * 2, 0.f);
  m_rotations_first_second.resize(num_gaussians * 2, vec4(0.f));
  m_scales_first_second.resize(num_gaussians * 2, vec3(0.f));
  m_sh_coefficient_0_first_second.resize(num_gaussians * 2, vec3(0.f));
  m_sh_coefficients_rest_first_second.resize(num_gaussians * 2 * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
  m_gaussian_steps.resize(num_gaussians, 0);
}

void AdamW::reset(int* indices, int num_reset) {
  size_t num_gaussians = m_gaussians->size();
  thrust::for_each(
    thrust::device_ptr<int>(indices),
    thrust::device_ptr<int>(indices) + num_reset,
    [
      means_first_second = m_means_first_second.data(),
      opacities_first_second = m_opacities_first_second.data(),
      rotations_first_second = m_rotations_first_second.data(),
      scales_first_second = m_scales_first_second.data(),
      sh_coefficient_0_first_second = m_sh_coefficient_0_first_second.data(),
      sh_coefficients_rest_first_second = m_sh_coefficients_rest_first_second.data(),
      gaussian_steps = m_gaussian_steps.data()
    ] __device__(int idx) {
      means_first_second[idx * 2] = vec3(0.f);
      means_first_second[idx * 2 + 1] = vec3(0.f);
      opacities_first_second[idx * 2] = 0.f;
      opacities_first_second[idx * 2 + 1] = 0.f;
      rotations_first_second[idx * 2] = vec4(0.f);
      rotations_first_second[idx * 2 + 1] = vec4(0.f);
      scales_first_second[idx * 2] = vec3(0.f);
      scales_first_second[idx * 2 + 1] = vec3(0.f);
      sh_coefficient_0_first_second[idx * 2] = vec3(0.f);
      sh_coefficient_0_first_second[idx * 2 + 1] = vec3(0.f);
      for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
        sh_coefficients_rest_first_second[idx * 2 * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
        sh_coefficients_rest_first_second[idx * 2 * (kMaxSphericalHarmonicsCoefficients - 1) + i + 1] = vec3(0.f);
      }
      gaussian_steps[idx] = 0;
    }
  );
}

void AdamW::reset_opacity() {
  // fxxk.
  thrust::fill(m_opacities_first_second.begin(), m_opacities_first_second.end(), 0.f);
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