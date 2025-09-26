#include <cuda/pipeline>
#include <thrust/execution_policy.h>
#include <cuda/barrier>
#include <nvtx3/nvtx3.hpp>
#include <cooperative_groups.h>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/simple_adam.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

namespace tinygs {

__device__ static __forceinline__ float lerp(float v0, float v1, float t) {
    return fmaf(t, v1, fmaf(-t, v0, v0));
}

__device__ static __forceinline__ vec3 lerp(vec3 v0, vec3 v1, float t) {
  return {
      lerp(v0.x, v1.x, t),
      lerp(v0.y, v1.y, t),
      lerp(v0.z, v1.z, t),
  };
}

__device__ static __forceinline__ vec4 lerp(vec4 v0, vec4 v1, float t) {
  return {
      lerp(v0.x, v1.x, t),
      lerp(v0.y, v1.y, t),
      lerp(v0.z, v1.z, t),
      lerp(v0.w, v1.w, t),
  };
}

struct NoDecay {
  template <typename T>
  __forceinline__ __device__ auto operator()(const T& /* theta */, const T &g) const noexcept {
    return g;
  }
};

template<typename T, typename DecayFunc = NoDecay>
__global__ static void adam_step(
    // Means
    T *__restrict__ thetas,
    T const *__restrict__ thetas_grad,
    T *__restrict__ thetas_first,
    T *__restrict__ thetas_second,
    // other
    SimpleAdamParameters adam_p,
    float lr,
    uint32_t num_gaussians,
    float gradient_scale,
    float bias_correction1,     // (1 - beta_1^t)
    float bias_correction2_sqrt, // sqrt(1 - beta_2^t)
    DecayFunc f = DecayFunc()
) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_gaussians) return;

  // Load
  T theta = thetas[idx];
  const T g = f(theta, gradient_scale * thetas_grad[idx]);
  T m = thetas_first[idx];
  T v = thetas_second[idx];
  const T g_sq = g * g;

  // Update biased first and second moment estimates
  m = lerp(m, g, 1.0f - adam_p.beta1);
  v = lerp(v, g_sq, 1.0f - adam_p.beta2);

  // Bias-corrected estimates
  const T m_hat = m / bias_correction1;
  const T denom = tinygs::sqrt(v) / bias_correction2_sqrt + adam_p.epsilon;

  // step
  theta -= m_hat * lr / denom;

  // write back updated params and moments
  thetas[idx] = theta;
  thetas_first[idx] = m;
  thetas_second[idx] = v;
}


template<uint nc, uint bsize=256, typename DecayFunc = NoDecay>
__global__ static void adam_step_with_shm(
    // Means
    float *__restrict__ thetas,
    float const *__restrict__ thetas_grad,
    float *__restrict__ thetas_first,
    float *__restrict__ thetas_second,
    // other
    SimpleAdamParameters adam_p,
    float lr,
    uint32_t num_gaussians,
    float gradient_scale,
    float bias_correction1,     // (1 - beta_1^t)
    float bias_correction2_sqrt, // sqrt(1 - beta_2^t)
    DecayFunc f = DecayFunc()
) {
  const auto block_first = blockIdx.x * blockDim.x;
  const auto local_idx = threadIdx.x;
  const auto idx = block_first + local_idx;
  auto block = cg::this_thread_block();
  uint gs_idx = idx / nc;
  uint coef_idx = idx % nc;
  if (idx >= num_gaussians) return;

  // Load
  float theta = thetas[idx];
  const float g = f(theta, gradient_scale * thetas_grad[idx]);
  float m = thetas_first[gs_idx * nc + coef_idx];
  float v = thetas_second[gs_idx * nc + coef_idx];
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
  thetas_first[gs_idx * nc + coef_idx] = m;
  thetas_second[gs_idx * nc + coef_idx] = v;
}



struct SimpleAdam_domain {
  static constexpr char const *name{"optim"};
};
using range = nvtx3::scoped_range_in<SimpleAdam_domain>;
using attr = nvtx3::event_attributes;
using regstr = nvtx3::registered_string_in<SimpleAdam_domain>;
using ncat = nvtx3::named_category_in<SimpleAdam_domain>;
static const nvtx3::rgb C_BLUE{0, 153, 255};
static const nvtx3::rgb C_ORANGE{255, 153, 0};
struct m_step {
  static constexpr char const *message{"simple_adam_step"};
};

void SimpleAdam::step(float scale) {

  NVTX3_FUNC_RANGE();
  const float gradient_scale = scale;  // This is the gradient scaler, not learning rate multiplier
  constexpr int block_size = 256;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("SimpleAdam::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("SimpleAdam::step: gaussians and gaussians_grad must have same size");
  }

  auto n = m_gaussians->size();
  const int grid = (n + block_size - 1) / block_size;

  m_global_steps++;
  const float bias_correction1 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(m_adam_params.beta1),
                     static_cast<double>(m_global_steps)));
  const float bias_correction2_sqrt = static_cast<float>(
      std::sqrt(1.0 - std::pow(static_cast<double>(m_adam_params.beta2),
                               static_cast<double>(m_global_steps))));

  {
    auto msg = regstr::get<m_step>();
    nvtx3::event_attributes attr(msg, nvtx3::payload{n});
    range range(attr);
    adam_step_with_shm<3><<<div_round_up<uint>(n * 3, block_size), block_size, 0, 0>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->means().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->means().data()),
      (float*) thrust::raw_pointer_cast(m_means_first.data()),
      (float*) thrust::raw_pointer_cast(m_means_second.data()),
      m_adam_params,
      m_params.means_lr,
      m_gaussians->size()*3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt
    );

    // Opacities
    adam_step_with_shm<1><<<div_round_up<uint>(n, block_size), block_size, 0, 0>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->opacities().data()),
      (float*) thrust::raw_pointer_cast(m_opacities_first.data()),
      (float*) thrust::raw_pointer_cast(m_opacities_second.data()),
      m_adam_params,
      m_params.opacities_lr,
      m_gaussians->size(),
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt
    );

    // Rotations
    adam_step_with_shm<4><<<div_round_up<uint>(n * 4, block_size), block_size, 0, 0>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->rotations().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->rotations().data()),
      (float*) thrust::raw_pointer_cast(m_rotations_first.data()),
      (float*) thrust::raw_pointer_cast(m_rotations_second.data()),
      m_adam_params,
      m_params.rotations_lr,
      m_gaussians->size() * 4,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt
    );

    // Scales
    adam_step_with_shm<3><<<div_round_up<uint>(n * 3, block_size), block_size, 0, 0>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->scales().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->scales().data()),
      (float*) thrust::raw_pointer_cast(m_scales_first.data()),
      (float*) thrust::raw_pointer_cast(m_scales_second.data()),
      m_adam_params,
      m_params.scales_lr,
      m_gaussians->size() * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt
    );

    // SH Coefficient 0
    adam_step_with_shm<3><<<div_round_up<uint>(n * 3, block_size), block_size, 0, 0>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficient_0().data()),
      (float*) thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
      (float*) thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
      m_adam_params,
      m_params.shs_lr,
      m_gaussians->size() * 3,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt
    );

    // SH Coefficients Rest
    const int sh_rest_size = m_gaussians->size() * (kMaxSphericalHarmonicsCoefficients - 1) * 3;
    adam_step_with_shm<3><<<div_round_up<uint>(sh_rest_size, block_size), block_size, 0, 0>>>(
      (float*) thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
      (float*) thrust::raw_pointer_cast(m_gaussians_grad->sh_coefficients_rest().data()),
      (float*) thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
      (float*) thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
      m_adam_params,
      m_params.shs_lr,
      sh_rest_size,
      gradient_scale,
      bias_correction1,
      bias_correction2_sqrt
    );
    maybe_sync(0);
  }
}


SimpleAdam::SimpleAdam(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
  OptimizerBase(gaussians, gaussians_grad) {
  // Resize and reset all internal buffers
  SimpleAdam::reset();
}

__global__ void copy_items(
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
  const vec3 * __restrict__ src_sh_coefficient_0_first,
  const vec3 * __restrict__ src_sh_coefficient_0_second,
  vec3 * __restrict__ dst_sh_coefficient_0_first,
  vec3 * __restrict__ dst_sh_coefficient_0_second,
  const vec3 * __restrict__ src_sh_coefficients_rest_first,
  const vec3 * __restrict__ src_sh_coefficients_rest_second,
  vec3 * __restrict__ dst_sh_coefficients_rest_first,
  vec3 * __restrict__ dst_sh_coefficients_rest_second,
  const int * __restrict__ mapping,
  int num_kept
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kept) return;

  int src_idx = mapping[idx];

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

  // Copy first/second moments for SH coefficient 0
  dst_sh_coefficient_0_first[idx] = src_sh_coefficient_0_first[src_idx];
  dst_sh_coefficient_0_second[idx] = src_sh_coefficient_0_second[src_idx];

  // Copy first/second moments for rest SH coefficients
  int src_rest_start = src_idx * (kMaxSphericalHarmonicsCoefficients - 1);
  int dst_rest_start = idx * (kMaxSphericalHarmonicsCoefficients - 1);
  for (int i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
    dst_sh_coefficients_rest_first[dst_rest_start + i] = src_sh_coefficients_rest_first[src_rest_start + i];
    dst_sh_coefficients_rest_second[dst_rest_start + i] = src_sh_coefficients_rest_second[src_rest_start + i];
  }
}

void SimpleAdam::remove(char* kept_flag, int num_kept) {
  // filters the gaussians' first second.
  size_t original_size = m_gaussians->size();
  thrust::device_vector<int> mapping(original_size); // mapping[idx] = original_idx

  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(original_size),
    mapping.begin(), [kept_flag] __device__ (int orig) { return static_cast<bool>(kept_flag[orig]); });

  thrust::device_vector<vec3> means_first(num_kept);
  thrust::device_vector<vec3> means_second(num_kept);
  thrust::device_vector<float> opacities_first(num_kept);
  thrust::device_vector<float> opacities_second(num_kept);
  thrust::device_vector<vec4> rotations_first(num_kept);
  thrust::device_vector<vec4> rotations_second(num_kept);
  thrust::device_vector<vec3> scales_first(num_kept);
  thrust::device_vector<vec3> scales_second(num_kept);
  thrust::device_vector<vec3> sh_coefficients_0_first(num_kept);
  thrust::device_vector<vec3> sh_coefficients_0_second(num_kept);
  thrust::device_vector<vec3> sh_coefficients_rest_first(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));
  thrust::device_vector<vec3> sh_coefficients_rest_second(num_kept * (kMaxSphericalHarmonicsCoefficients - 1));

  const int grid = (num_kept + 255) / 256;
  copy_items<<<grid, 256>>>(
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
      thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
      thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
      thrust::raw_pointer_cast(sh_coefficients_0_first.data()),
      thrust::raw_pointer_cast(sh_coefficients_0_second.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
      thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest_first.data()),
      thrust::raw_pointer_cast(sh_coefficients_rest_second.data()),
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
  m_sh_coefficient_0_first = std::move(sh_coefficients_0_first);
  m_sh_coefficient_0_second = std::move(sh_coefficients_0_second);
  m_sh_coefficients_rest_first = std::move(sh_coefficients_rest_first);
  m_sh_coefficients_rest_second = std::move(sh_coefficients_rest_second);
  CUDA_CHECK_THROW(cudaDeviceSynchronize()); CUDA_CHECK_THROW(cudaGetLastError());
}

__global__ static void duplicate_optimizer_state_kernel(
  // Means
  vec3* __restrict__ means_first,
  vec3* __restrict__ means_second,
  // Opacities
  float* __restrict__ opacities_first,
  float* __restrict__ opacities_second,
  // Rotations
  vec4* __restrict__ rotations_first,
  vec4* __restrict__ rotations_second,
  // Scales
  vec3* __restrict__ scales_first,
  vec3* __restrict__ scales_second,
  // Spherical Harmonics
  vec3* __restrict__ sh_coefficient_0_first,
  vec3* __restrict__ sh_coefficient_0_second,
  vec3* __restrict__ sh_coefficients_rest_first,
  vec3* __restrict__ sh_coefficients_rest_second,
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
  means_first[dst_idx] = means_first[src_idx] * kHalf;
  means_second[dst_idx] = means_second[src_idx] * kQuarter;

  // Copy opacities first and second moments
  opacities_first[dst_idx] = opacities_first[src_idx] * kHalf;
  opacities_second[dst_idx] = opacities_second[src_idx] * kQuarter;

  // Copy rotations first and second moments
  rotations_first[dst_idx] = rotations_first[src_idx] * kHalf;
  rotations_second[dst_idx] = rotations_second[src_idx] * kQuarter;

  // Copy scales first and second moments
  scales_first[dst_idx] = scales_first[src_idx] * kHalf;
  scales_second[dst_idx] = scales_second[src_idx] * kQuarter;

  // Copy sh_coefficient_0 first and second moments
  sh_coefficient_0_first[dst_idx] = sh_coefficient_0_first[src_idx] * kHalf;
  sh_coefficient_0_second[dst_idx] = sh_coefficient_0_second[src_idx] * kQuarter;

  // Copy sh_coefficients_rest first and second moments
  for (uint32_t i = 0; i < num_sh_rest_per_gaussian; i++) {
    const int src_sh_idx = src_idx * num_sh_rest_per_gaussian + i;
    const int dst_sh_idx = dst_idx * num_sh_rest_per_gaussian + i;
    sh_coefficients_rest_first[dst_sh_idx] = sh_coefficients_rest_first[src_sh_idx] * kHalf;
    sh_coefficients_rest_second[dst_sh_idx] = sh_coefficients_rest_second[src_sh_idx] * kQuarter;
  }

}

void SimpleAdam::duplicate(int* indices, int* new_indices, int num_duplicate) {
  if (num_duplicate == 0) return;
  const uint32_t num_sh_rest_per_gaussian = kMaxSphericalHarmonicsCoefficients - 1;
  // Resize vectors to accommodate duplicated gaussians
  m_means_first.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_means_second.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_opacities_first.resize(m_gaussians->size(), 0.f);
  m_opacities_second.resize(m_gaussians->size(), 0.f);
  m_rotations_first.resize(m_gaussians->size(), vec4(0.f, 0.f, 0.f, 0.f));
  m_rotations_second.resize(m_gaussians->size(), vec4(0.f, 0.f, 0.f, 0.f));
  m_scales_first.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_scales_second.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_sh_coefficient_0_first.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_sh_coefficient_0_second.resize(m_gaussians->size(), vec3(0.f, 0.f, 0.f));
  m_sh_coefficients_rest_first.resize(m_gaussians->size() * num_sh_rest_per_gaussian, vec3(0.f, 0.f, 0.f));
  m_sh_coefficients_rest_second.resize(m_gaussians->size() * num_sh_rest_per_gaussian, vec3(0.f, 0.f, 0.f));

  // TODO: This design does not provide better result. Why?
  // const int grid = (num_duplicate + 255) / 256;
  // duplicate_optimizer_state_kernel<<<grid, 256>>>(
  //   thrust::raw_pointer_cast(m_means_first.data()),
  //   thrust::raw_pointer_cast(m_means_second.data()),
  //   thrust::raw_pointer_cast(m_opacities_first.data()),
  //   thrust::raw_pointer_cast(m_opacities_second.data()),
  //   thrust::raw_pointer_cast(m_rotations_first.data()),
  //   thrust::raw_pointer_cast(m_rotations_second.data()),
  //   thrust::raw_pointer_cast(m_scales_first.data()),
  //   thrust::raw_pointer_cast(m_scales_second.data()),
  //   thrust::raw_pointer_cast(m_sh_coefficient_0_first.data()),
  //   thrust::raw_pointer_cast(m_sh_coefficient_0_second.data()),
  //   thrust::raw_pointer_cast(m_sh_coefficients_rest_first.data()),
  //   thrust::raw_pointer_cast(m_sh_coefficients_rest_second.data()),
  //   indices,
  //   new_indices,
  //   num_duplicate,
  //   num_sh_rest_per_gaussian
  // );
}

void SimpleAdam::reset() {
  size_t num_gaussians = m_gaussians->size();
  m_global_steps = 0;

  m_means_first.resize(num_gaussians, vec3(0.f));
  m_means_second.resize(num_gaussians, vec3(0.f));
  m_opacities_first.resize(num_gaussians, 0.f);
  m_opacities_second.resize(num_gaussians, 0.f);
  m_rotations_first.resize(num_gaussians, vec4(0.f));
  m_rotations_second.resize(num_gaussians, vec4(0.f));
  m_scales_first.resize(num_gaussians, vec3(0.f));
  m_scales_second.resize(num_gaussians, vec3(0.f));
  m_sh_coefficient_0_first.resize(num_gaussians, vec3(0.f));
  m_sh_coefficient_0_second.resize(num_gaussians, vec3(0.f));
  m_sh_coefficients_rest_first.resize(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
  m_sh_coefficients_rest_second.resize(num_gaussians * (kMaxSphericalHarmonicsCoefficients - 1), vec3(0.f));
}

void SimpleAdam::reset(int* indices, int num_reset) {
  size_t num_gaussians = m_gaussians->size();
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
      sh_coefficient_0_first = m_sh_coefficient_0_first.data(),
      sh_coefficient_0_second = m_sh_coefficient_0_second.data(),
      sh_coefficients_rest_first = m_sh_coefficients_rest_first.data(),
      sh_coefficients_rest_second = m_sh_coefficients_rest_second.data()
    ] __device__(int idx) {
      means_first[idx] = vec3(0.f);
      means_second[idx] = vec3(0.f);
      opacities_first[idx] = 0.f;
      opacities_second[idx] = 0.f;
      rotations_first[idx] = vec4(0.f);
      rotations_second[idx] = vec4(0.f);
      scales_first[idx] = vec3(0.f);
      scales_second[idx] = vec3(0.f);
      sh_coefficient_0_first[idx] = vec3(0.f);
      sh_coefficient_0_second[idx] = vec3(0.f);
      for (uint32_t i = 0; i < kMaxSphericalHarmonicsCoefficients - 1; i++) {
        sh_coefficients_rest_first[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
        sh_coefficients_rest_second[idx * (kMaxSphericalHarmonicsCoefficients - 1) + i] = vec3(0.f);
      }
    }
  );
}

void SimpleAdam::reset_opacity() {
  // fxxk.
  thrust::fill(m_opacities_first.begin(), m_opacities_first.end(), 0.f);
  thrust::fill(m_opacities_second.begin(), m_opacities_second.end(), 0.f);
}

void SimpleAdam::set_params(const json& config) {
  // Update base optimizer parameters
  OptimizerBase::set_params(config);

  // Update SimpleAdam-specific parameters
  m_adam_params.from_json(config);
}

json SimpleAdam::get_params() const {
  // Get base optimizer parameters
  json params = OptimizerBase::get_params();

  // Add type for reflection
  params["type"] = "adam";

  // Add SimpleAdam-specific parameters
  json SimpleAdam_params = m_adam_params.to_json();

  // Merge the two JSON objects
  for (auto& [key, value] : SimpleAdam_params.items()) {
    params[key] = value;
  }

  return params;
}

json SimpleAdamParameters::to_json() const {
  json j;
  j["beta1"] = beta1;
  j["beta2"] = beta2;
  j["epsilon"] = epsilon;
  // j["enable_adabound"] = enable_adabound; // Removed enable_adabound
  return j;
}

void SimpleAdamParameters::from_json(const json& config) {
  if (config.contains("beta1")) {
    beta1 = config.at("beta1").get<float>();
  }
  if (config.contains("beta2")) {
    beta2 = config.at("beta2").get<float>();
  }
  if (config.contains("epsilon")) {
    epsilon = config.at("epsilon").get<float>();
  }
  if (config.contains("enable_adabound")) {
    // Removed enable_adabound
  }
}

} // namespace tinygs