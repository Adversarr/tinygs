#include "tinygs/optim/adamw.hpp"
#include "tinygs/cuda/common_device.cuh"

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
  float relative_weight_decay, // l2
  float absolute_weight_decay, // l1
  const float& weight_clipping_magnitude,
  const float& gradient_clipping_magnitude,
  float loss_scale,
  const float& lower_lr_bound,
  const float& upper_lr_bound,
  const float& this_lr_scale
) {
  gradient *= loss_scale;
  if (gradient_clipping_magnitude != 0.0f) {
    gradient = copysign(min(abs(gradient), gradient_clipping_magnitude), gradient);
  }

  const Elem gradient_sq = gradient * gradient;
  first_moment = beta1 * first_moment + (1 - beta1) * gradient;
  second_moment = beta2 * second_moment + (1 - beta2) * gradient_sq;

  // Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
  learning_rate *= this_lr_scale;

  // Follow AdaBound paradigm
  const Elem effective_learning_rate
      = min(max(learning_rate / (sqrt(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

  relative_weight_decay *= learning_rate;
  absolute_weight_decay *= learning_rate;

  const Elem decayed_weight = (1 - relative_weight_decay) * weight - copysign(absolute_weight_decay, weight);
  Elem new_weight = decayed_weight - effective_learning_rate * first_moment;

  // if (weight_clipping_magnitude != 0.0f) {
  //   new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
  // }

  weight = new_weight;
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
  float loss_scale
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians) return;

  // accumulate all the gradients, to cull out zero gradient gaussians
  float sh_grad_norm = sum(abs(sh_coefficient_0_grad[idx]));
  for (int i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
    sh_grad_norm += sum(abs(sh_coefficients_rest_grad[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i]));
  }
  const float grad_norm_1 = (
    sum(abs(means_grad[idx])) + abs(opacities_grad[idx]) +
    sum(abs(rotations_grad[idx])) +
    sum(abs(scales_grad[idx])) +
    sh_grad_norm
  );
  if (grad_norm_1 == 0 && general_p.skip_zero_grad) return;

  const float beta1 = adam_p.beta1;
  const float beta2 = adam_p.beta2;

  // actually perform the optimization for this gaussian
  const auto this_step = gaussian_steps[idx] += 1;
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
    const vec3& grad = means_grad[idx];
    vec3& first_moment = means_first_second[idx * 2];
    vec3& second_moment = means_first_second[idx * 2 + 1];
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.means_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.means_l2, general_p.means_l1,
      0.0f,
      general_p.max_grad_1,
      loss_scale,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // opacities
    float& val = opacities[idx];
    const float& grad = opacities_grad[idx];
    float& first_moment = opacities_first_second[idx * 2];
    float& second_moment = opacities_first_second[idx * 2 + 1];
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.opacities_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.opacities_l2, general_p.opacities_l1,
      0.0f,
      general_p.max_grad_1,
      loss_scale,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // rotations
    vec4& val = rotations[idx];
    const vec4& grad = rotations_grad[idx];
    vec4& first_moment = rotations_first_second[idx * 2];
    vec4& second_moment = rotations_first_second[idx * 2 + 1];
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.rotations_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.rotations_l2, general_p.rotations_l1,
      0.0f,
      general_p.max_grad_1,
      loss_scale,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // scales
    vec3& val = scales[idx];
    const vec3& grad = scales_grad[idx];
    vec3& first_moment = scales_first_second[idx * 2];
    vec3& second_moment = scales_first_second[idx * 2 + 1];
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.scales_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.scales_l2, general_p.scales_l1,
      0.0f,
      general_p.max_grad_1,
      loss_scale,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // spherical harmonics - 0th coefficient
    vec3& val = sh_coefficient_0[idx];
    const vec3& grad = sh_coefficient_0_grad[idx];
    vec3& first_moment = sh_coefficient_0_first_second[idx * 2];
    vec3& second_moment = sh_coefficient_0_first_second[idx * 2 + 1];
    adam_step_func(
      val, grad, first_moment, second_moment,
      general_p.shs_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.shs_l2, general_p.shs_l1,
      0.0f,
      general_p.max_grad_1,
      loss_scale,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // spherical harmonics - rest coefficients
    int start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
    int end = start + (kMaxSphericalHarmonicsCoefficients - 1);
    for (int i = start; i < end; i++) {
      vec3& val = sh_coefficients_rest[i];
      const vec3& grad = sh_coefficients_rest_grad[i];
      vec3& first_moment = sh_coefficients_rest_first_second[i * 2];
      vec3& second_moment = sh_coefficients_rest_first_second[i * 2 + 1];
      adam_step_func(
        val, grad, first_moment, second_moment,
        general_p.shs_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
        general_p.shs_l2, general_p.shs_l1,
        0.0f,
        general_p.max_grad_1,
        loss_scale,
        lower_lr_bound,
        upper_lr_bound,
        this_lr_scale
      );
    }
  }
}

void AdamW::step(float scale) {
  const float loss_scale = scale;
  const int grid = (m_gaussians->size() + 255) / 256;

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
    loss_scale
  );
}

AdamW::AdamW(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad,
      const AdamWParameters& params):
  OptimizerBase(gaussians, gaussians_grad), m_adam_params(params) {
  // Resize and reset all internal buffers
  AdamW::reset();
}

template <typename T> static void filter(
  thrust::device_vector<T>& in_out,
  char* kept_flag,
  int num_kept,
  cudaStream_t stream
) {
  T* d_in = thrust::raw_pointer_cast(in_out.data());
  size_t num_items = in_out.size();
  GPUBuffer<int> d_num_selected_out(stream, 1);
  size_t temp_storage_bytes = 0;
  CUDA_CHECK_THROW(cub::DeviceSelect::Flagged( //
      nullptr, temp_storage_bytes,             //
      d_in, kept_flag, d_num_selected_out.data(), num_items, stream));
  GPUBuffer<int> d_temp_storage(stream, temp_storage_bytes);
  CUDA_CHECK_THROW(cub::DeviceSelect::Flagged(           //
      (void *)d_temp_storage.data(), temp_storage_bytes, //
      d_in, kept_flag, d_num_selected_out.data(), num_items, stream));
  in_out.resize(num_kept);
}

void AdamW::remove(char* kept_flag, int num_kept) {
  // filters the gaussians' first second.
  int num_items = m_gaussians->size();

  filter(m_means_first_second, kept_flag, num_kept, 0);
  filter(m_opacities_first_second, kept_flag, num_kept, 0);
  filter(m_rotations_first_second, kept_flag, num_kept, 0);
  filter(m_scales_first_second, kept_flag, num_kept, 0);
  filter(m_sh_coefficient_0_first_second, kept_flag, num_kept, 0);
  // TODO: this is not correct.
  filter(m_sh_coefficients_rest_first_second, kept_flag, num_kept, 0);
  filter(m_gaussian_steps, kept_flag, num_kept, 0);
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

  // Copy means first and second moments
  means_first_second[dst_idx * 2] = means_first_second[src_idx * 2];
  means_first_second[dst_idx * 2 + 1] = means_first_second[src_idx * 2 + 1];

  // Copy opacities first and second moments
  opacities_first_second[dst_idx * 2] = opacities_first_second[src_idx * 2];
  opacities_first_second[dst_idx * 2 + 1] = opacities_first_second[src_idx * 2 + 1];

  // Copy rotations first and second moments
  rotations_first_second[dst_idx * 2] = rotations_first_second[src_idx * 2];
  rotations_first_second[dst_idx * 2 + 1] = rotations_first_second[src_idx * 2 + 1];

  // Copy scales first and second moments
  scales_first_second[dst_idx * 2] = scales_first_second[src_idx * 2];
  scales_first_second[dst_idx * 2 + 1] = scales_first_second[src_idx * 2 + 1];

  // Copy sh_coefficient_0 first and second moments
  sh_coefficient_0_first_second[dst_idx * 2] = sh_coefficient_0_first_second[src_idx * 2];
  sh_coefficient_0_first_second[dst_idx * 2 + 1] = sh_coefficient_0_first_second[src_idx * 2 + 1];

  // Copy sh_coefficients_rest first and second moments
  for (uint32_t i = 0; i < num_sh_rest_per_gaussian; i++) {
    const int src_sh_idx = src_idx * num_sh_rest_per_gaussian + i;
    const int dst_sh_idx = dst_idx * num_sh_rest_per_gaussian + i;
    sh_coefficients_rest_first_second[dst_sh_idx * 2] = sh_coefficients_rest_first_second[src_sh_idx * 2];
    sh_coefficients_rest_first_second[dst_sh_idx * 2 + 1] = sh_coefficients_rest_first_second[src_sh_idx * 2 + 1];
  }

  // Copy step count
  gaussian_steps[dst_idx] = gaussian_steps[src_idx];
}

void AdamW::duplicate(int* /*indices*/, int* /*new_indices*/, int num_duplicate) {
  if (num_duplicate == 0) return;
  reset(); return;

  const uint32_t num_sh_rest_per_gaussian = kMaxSphericalHarmonicsCoefficients - 1;

  // Resize vectors to accommodate duplicated gaussians
  m_means_first_second.resize(m_gaussians->size() * 2, vec3(0.f));
  m_opacities_first_second.resize(m_gaussians->size() * 2, 0.f);
  m_rotations_first_second.resize(m_gaussians->size() * 2, vec4(0.f));
  m_scales_first_second.resize(m_gaussians->size() * 2, vec3(0.f));
  m_sh_coefficient_0_first_second.resize(m_gaussians->size() * 2, vec3(0.f));
  m_sh_coefficients_rest_first_second.resize(m_gaussians->size() * 2 * num_sh_rest_per_gaussian, vec3(0.f));
  m_gaussian_steps.resize(m_gaussians->size(), 0);

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
  CUDA_CHECK_THROW(cudaDeviceSynchronize());
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

}