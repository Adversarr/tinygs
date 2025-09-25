#include <cuda/pipeline>
#include <thrust/execution_policy.h>

#include <nvtx3/nvtx3.hpp>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adamw.hpp"

namespace tinygs {

__device__ void adam_step_func(
  float& weight,
  float gradient,
  float& first_moment,
  float& second_moment,
  float learning_rate,
  const float& beta1,
  const float& beta2,
  const float& epsilon,
  const float& gradient_clipping_magnitude,
  const float& lower_lr_bound,
  const float& upper_lr_bound
) {
  if (gradient_clipping_magnitude != 0.0f) {
    gradient = copysign(min(abs(gradient), gradient_clipping_magnitude), gradient);
  }

  const float gradient_sq = gradient * gradient;
  first_moment = beta1 * first_moment + (1 - beta1) * gradient;
  second_moment = beta2 * second_moment + (1 - beta2) * gradient_sq;

  // // Follow AdaBound paradigm
  const float effective_learning_rate
      = fmin(fmax(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

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
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  const uint block_leader_thread_idx = blockIdx.x * blockDim.x;
  const auto local_idx = threadIdx.x;
  const auto idx = block_leader_thread_idx + local_idx;
  const uint this_block_range = ::min(block.size(), num_gaussians - block_leader_thread_idx);
  constexpr size_t stages_count = 2; // Pipeline with 2 stage
  // note: we cannot return, the last thread will be used to enable barrier.
  bool enable = idx < num_gaussians;

  /* == common settings ==  */
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      stages_count
  > shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state);
  // 2 stage for async, block.size() * 4      *   4B * 4               * 2
  //                                   float4 | v, grad, first, second | stage
  // it should < 4096B, otherwise will limit the occupancy.
  // => 4096/128 = 32 = block.size() ? This is too small. (One warp per SM)
  // However, if we have better GPUs, this could be much much much larger!
  extern __shared__ float4 shared[];
  float4* const shared_val = shared;
  float4* const shared_grad = shared + block.size(); // float4 * bsize
  float4* const shared_momentum = shared + 2 * block.size();
  const size_t shared_offset[stages_count] = {0, 4 * block.size()};

  const double beta1 = adam_p.beta1;
  const double beta2 = adam_p.beta2;
  const double epsilon = adam_p.epsilon;
  const float max_grad_1 = general_p.max_grad_1;

  uint curr_stage_store = 0;
  uint curr_stage_compute = 0;
  // AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
  float lower_lr_bound = 0;
  float upper_lr_bound = std::numeric_limits<float>::max();

  // TODO: typically very large, will lose accuracy.
  const float inv_n = 1.0f / num_gaussians;

  // actually perform the optimization for this gaussian
  uint this_step = 0;
  float this_lr_scale = 0;

#define PREFETCH_SHM(type, field_name)                                                                               \
  pipeline.producer_acquire();                                                                                       \
  {uint cpy_bytes = div_round_up<uint>(sizeof(type) * this_block_range, 16u) * 16u;                                         \
  uint cpy_bytes2 = div_round_up<uint>(sizeof(type) * this_block_range * 2, 16u) * 16u;                                    \
  cuda::memcpy_async(block, shared_val + shared_offset[curr_stage_store], (const float4*) ((field_name) + block_leader_thread_idx),    \
                     cpy_bytes, pipeline);                                                                           \
  cuda::memcpy_async(block, shared_grad + shared_offset[curr_stage_store],                                           \
                     (const float4*) ( (field_name##_grad) + block_leader_thread_idx), cpy_bytes, pipeline);      \
  cuda::memcpy_async(block, shared_momentum + shared_offset[curr_stage_store],                                       \
                     (const float4*) ((field_name##_first_second) + 2 * block_leader_thread_idx), cpy_bytes2, \
                     pipeline);                                                                                      \
  pipeline.producer_commit();                                                                                        }\
  curr_stage_store = (curr_stage_store + 1) % stages_count

  PREFETCH_SHM(float3, means);
  PREFETCH_SHM(float, opacities);

  const float beta1_f = (float)beta1;
  const float beta2_f = (float)beta2;
  const float epsilon_f = (float)epsilon;

#define apply_(v, g, f, s, lr) adam_step_func((v), (g), (f), (s), (lr), beta1_f, beta2_f, epsilon_f, max_grad_1, lower_lr_bound, upper_lr_bound)

  // Use mean
  {
    vec3* pval = reinterpret_cast<vec3*>(shared_val + shared_offset[curr_stage_compute]);
    const vec3* pgrad = reinterpret_cast<const vec3*>(shared_grad + shared_offset[curr_stage_compute]);
    vec3* pmom = reinterpret_cast<vec3*>(shared_momentum + shared_offset[curr_stage_compute]);
    pipeline.consumer_wait();
    if (enable) {
      vec3 val = pval[local_idx];
      vec3 grad = pgrad[local_idx];
      vec3 first = pmom[local_idx * 2];
      vec3 second = pmom[local_idx * 2 + 1];

      if (sum(abs(grad)) == 0) {
        enable = false;
      } 

      if (enable) {
        this_step = ++gaussian_steps[idx];
        this_lr_scale = ::sqrt(1. - ::pow(beta2, static_cast<double>(this_step))) /
                              (1. - ::pow(beta1, static_cast<double>(this_step)));
        const float lr = (general_p.means_lr * global_lr) * this_lr_scale;
        apply_(val.x, grad.x, first.x, second.x, lr);
        apply_(val.y, grad.y, first.y, second.y, lr);
        apply_(val.z, grad.z, first.z, second.z, lr);
        means[idx] = val;
        means_first_second[idx * 2] = first;
        means_first_second[idx * 2 + 1] = second;
      }
    }
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
    PREFETCH_SHM(float4, rotations); // means is used, we load rotations now.
  }

  { // opacities
    pipeline.consumer_wait();
    float* pval = reinterpret_cast<float*>(shared_val + shared_offset[curr_stage_compute]);
    const float* pgrad = reinterpret_cast<const float*>(shared_grad + shared_offset[curr_stage_compute]);
    float* pmom = reinterpret_cast<float*>(shared_momentum + shared_offset[curr_stage_compute]);
    float val = pval[local_idx];
    float grad = pgrad[local_idx] * gradient_scale + (general_p.opacities_l1 * activate_opacity_deriv(val)) * inv_n;
    float first = pmom[local_idx * 2];
    float second = pmom[local_idx * 2 + 1];
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
    PREFETCH_SHM(vec3, scales);
    if (enable) {
      const float lr =  general_p.opacities_lr * global_lr * this_lr_scale;
      apply_(val, grad, first, second, lr);
      opacities[idx] = val;
      opacities_first_second[idx * 2] = first;
      opacities_first_second[idx * 2 + 1] = second;
    }
  }

  { // rotations
    pipeline.consumer_wait();
    vec4* pval = reinterpret_cast<vec4*>(shared_val + shared_offset[curr_stage_compute]);
    const vec4* pgrad = reinterpret_cast<const vec4*>(shared_grad + shared_offset[curr_stage_compute]);
    vec4* pmom = reinterpret_cast<vec4*>(shared_momentum + shared_offset[curr_stage_compute]);
    if (enable) {
      vec4 val = pval[local_idx];
      vec4 grad = pgrad[local_idx];
      vec4 first = pmom[local_idx * 2];
      vec4 second = pmom[local_idx * 2 + 1];
      const float lr = general_p.rotations_lr * global_lr * this_lr_scale;
      apply_(val.x, grad.x, first.x, second.x, lr);
      apply_(val.y, grad.y, first.y, second.y, lr);
      apply_(val.z, grad.z, first.z, second.z, lr);
      apply_(val.w, grad.w, first.w, second.w, lr);
      rotations[idx] = val;
      rotations_first_second[idx * 2] = first;
      rotations_first_second[idx * 2 + 1] = second;
    }
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
    PREFETCH_SHM(vec3, sh_coefficient_0);
  }

  { // scales
    pipeline.consumer_wait();
    vec3* pval = reinterpret_cast<vec3*>(shared_val + shared_offset[curr_stage_compute]);
    const vec3* pgrad = reinterpret_cast<const vec3*>(shared_grad + shared_offset[curr_stage_compute]);
    vec3* pmom = reinterpret_cast<vec3*>(shared_momentum + shared_offset[curr_stage_compute]);
    if (enable) {
      vec3 val = pval[local_idx];
      vec3 grad = pgrad[local_idx];
      vec3 first = pmom[local_idx * 2];
      vec3 second = pmom[local_idx * 2 + 1];
      const float lr = general_p.scales_lr * global_lr * this_lr_scale;
      apply_(val.x, grad.x, first.x, second.x, lr);
      apply_(val.y, grad.y, first.y, second.y, lr);
      apply_(val.z, grad.z, first.z, second.z, lr);
      scales[idx] = val;
      scales_first_second[idx * 2] = first;
      scales_first_second[idx * 2 + 1] = second;
    }
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
  }

  { // spherical harmonics - 0th coefficient
    pipeline.consumer_wait();
    vec3* pval = reinterpret_cast<vec3*>(shared_val + shared_offset[curr_stage_compute]);
    const vec3* pgrad = reinterpret_cast<const vec3*>(shared_grad + shared_offset[curr_stage_compute]);
    vec3* pmom = reinterpret_cast<vec3*>(shared_momentum + shared_offset[curr_stage_compute]);
    vec3 val = pval[local_idx];
    vec3 grad = pgrad[local_idx];
    vec3 first = pmom[local_idx * 2];
    vec3 second = pmom[local_idx * 2 + 1];
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();

    if (enable) {
      const float lr = general_p.shs_lr * global_lr * this_lr_scale;
      apply_(val.x, grad.x, first.x, second.x, lr);
      apply_(val.y, grad.y, first.y, second.y, lr);
      apply_(val.z, grad.z, first.z, second.z, lr);
      sh_coefficient_0[idx] = val;
      sh_coefficient_0_first_second[idx * 2] = first;
      sh_coefficient_0_first_second[idx * 2 + 1] = second;
    }
  }

  if (enable) { // spherical harmonics - rest coefficients
    // NOTE: They use 1/20 LR w.r.t. sh0
    int start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
    int end = start + (kMaxSphericalHarmonicsCoefficients - 1);
    for (int i = start; i < end; i++) {
      vec3& val = sh_coefficients_rest[i];
      const vec3 grad = sh_coefficients_rest_grad[i] * gradient_scale;
      vec3& first_moment = sh_coefficients_rest_first_second[i * 2];
      vec3& second_moment = sh_coefficients_rest_first_second[i * 2 + 1];
      const float lr = general_p.shs_lr * global_lr * this_lr_scale * 0.05f;
      if (enable) apply_(val.x, grad.x, first_moment.x, second_moment.x, lr);
      if (enable) apply_(val.y, grad.y, first_moment.y, second_moment.y, lr);
      if (enable) apply_(val.z, grad.z, first_moment.z, second_moment.z, lr);
    }
  }
}

void AdamW::step(float scale) {
  NVTX3_FUNC_RANGE();
  const float gradient_scale = scale;  // This is the gradient scaler, not learning rate multiplier
  constexpr int block_size = 256;
  const int grid = (m_gaussians->size() + block_size - 1) / block_size;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("AdamW::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("AdamW::step: gaussians and gaussians_grad must have same size");
  }

  auto expected_shm = 4 * block_size * sizeof(float) * 4 * 2;

  launch_gaussian_adam_step_SoA<<<grid, block_size, expected_shm, 0>>>(
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

  CUDA_CHECK_THROW(cudaDeviceSynchronize());
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