#include "tinygs/optim/adamw.hpp"
#include "tinygs/cuda/common_device.cuh"

namespace tinygs {

template <typename Elem>
__forceinline__ __device__ void adam_step_func(
  Elem& weight,
  Elem gradient,
  Elem& first_moment,
  Elem& second_moment,
  uint32_t current_step,
  float learning_rate,
  const float& beta1,
  const float& beta2,
  const float& epsilon,
  float relative_weight_decay, // l2
  float absolute_weight_decay, // l1
  const float& weight_clipping_magnitude,
  const float& gradient_clipping_magnitude,
  float inv_loss_scale,
  const float& lower_lr_bound,
  const float& upper_lr_bound,
  const float& this_lr_scale
) {
  gradient *= inv_loss_scale;
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
  vec3* __restrict__ sh_coefficients,
  const vec3* __restrict__ sh_coefficients_grad,
  vec3* __restrict__ sh_coefficients_first_second,
  // other
  uint32_t* __restrict__ gaussian_steps,
  AdamWParameters adam_p,
  GaussianOptimizationParams general_p,
  uint32_t num_gaussians,
  float inv_loss_scale
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians) return;

  // accumulate all the gradients, to cull out zero gradient gaussians
  const float grad_norm_1 = (
    sum(abs(means_grad[idx])) + abs(opacities_grad[idx]) +
    sum(abs(rotations_grad[idx])) +
    sum(abs(scales_grad[idx])) +
    sum(abs(sh_coefficients_grad[idx]))
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
      this_step,
      general_p.means_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.means_l2, general_p.means_l1,
      0.0f,
      general_p.max_grad_1,
      inv_loss_scale,
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
      this_step,
      general_p.opacities_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.opacities_l2, general_p.opacities_l1,
      0.0f,
      general_p.max_grad_1,
      inv_loss_scale,
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
      this_step,
      general_p.rotations_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.rotations_l2, general_p.rotations_l1,
      0.0f,
      general_p.max_grad_1,
      inv_loss_scale,
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
      this_step,
      general_p.scales_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
      general_p.scales_l2, general_p.scales_l1,
      0.0f,
      general_p.max_grad_1,
      inv_loss_scale,
      lower_lr_bound,
      upper_lr_bound,
      this_lr_scale
    );
  }

  { // spherical harmonics
    int start = idx * kMaxSphericalHarmonicsCoefficients;
    int end = start + kMaxSphericalHarmonicsCoefficients;
    for (int i = start; i < end; i++) {
      vec3& val = sh_coefficients[i];
      const vec3& grad = sh_coefficients_grad[i];
      vec3& first_moment = sh_coefficients_first_second[i * 2];
      vec3& second_moment = sh_coefficients_first_second[i * 2 + 1];
      adam_step_func(
        val, grad, first_moment, second_moment,
        this_step,
        general_p.shs_lr, adam_p.beta1, adam_p.beta2, adam_p.epsilon,
        general_p.shs_l2, general_p.shs_l1,
        0.0f,
        general_p.max_grad_1,
        inv_loss_scale,
        lower_lr_bound,
        upper_lr_bound,
        this_lr_scale
      );
    }
  }
}

void AdamW::step(float scale) {
  const float inv_loss_scale = 1.0f / scale;
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
    thrust::raw_pointer_cast(m_gaussians->sh_coefficients().data()),
    thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients().data()),
    thrust::raw_pointer_cast(m_sh_coefficients_first_second.data()),
    thrust::raw_pointer_cast(m_gaussian_steps.data()),
    m_adam_params,
    m_params,
    m_gaussians->size(),
    inv_loss_scale
  );
}

void AdamW::reset() {
  size_t num_gaussians = m_gaussians->size();

  m_means_first_second.resize(num_gaussians * 2, vec3(0.f));
  m_opacities_first_second.resize(num_gaussians * 2, 0.f);
  m_rotations_first_second.resize(num_gaussians * 2, vec4(0.f));
  m_scales_first_second.resize(num_gaussians * 2, vec3(0.f));
  m_sh_coefficients_first_second.resize(num_gaussians * 2 * kMaxSphericalHarmonicsCoefficients, vec3(0.f));
  m_gaussian_steps.resize(num_gaussians, 0);
}

}