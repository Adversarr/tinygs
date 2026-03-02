#include <cfloat>
#include <cuda/pipeline>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <nvtx3/nvtx3.hpp>
#include <cooperative_groups.h>
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/optim/adamw.hpp"

namespace tinygs {

__device__ static __forceinline__ float lerp(float v0, float v1, float t) {
    return fmaf(t, v1, fmaf(-t, v0, v0));
}

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
    gradient = copysignf(fminf(fabsf(gradient), gradient_clipping_magnitude), gradient);
  }

  const float gradient_sq = gradient * gradient;
  first_moment = lerp(first_moment, gradient, 1 - beta1);      // exp_avg
  second_moment = lerp(second_moment, gradient_sq, 1 - beta2); // exp_avg_sq
  // Follow AdaBound paradigm (numerically stable)
  {
    const float denom = sqrtf(second_moment) + epsilon;
    const float safe_denom = (denom == 0.0f) ? 1e-30f : denom;
    const float eff_lr_d = learning_rate / safe_denom;
    const float effective_learning_rate = fminf(fmaxf(eff_lr_d, lower_lr_bound), upper_lr_bound);
    weight -= effective_learning_rate * first_moment;
  }
}

__global__ void launch_gaussian_adam_step_SoA_ref2(
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
  // other
  uint32_t* __restrict__ gaussian_steps,
  AdamWParameters adam_p,
  GaussianOptimizationParams general_p,
  uint32_t num_gaussians,
  const float gradient_scale,
  const float global_lr
) {
  auto block = cooperative_groups::this_thread_block();
  const unsigned int block_leader_thread_idx = blockIdx.x * blockDim.x;
  const unsigned int local_idx = threadIdx.x;
  const unsigned int idx = block_leader_thread_idx + local_idx;
  // Fix: Use signed int to avoid unsigned underflow
  int remaining = static_cast<int>(num_gaussians) - static_cast<int>(block_leader_thread_idx);
  remaining = max(0, remaining);
  const unsigned int this_block_range = min(blockDim.x, static_cast<unsigned int>(remaining));
  constexpr size_t stages_count = 2;  // Pipeline with 2 stages
  bool enable = idx < num_gaussians;
  /* == common settings == */
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      stages_count
  > shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state);
  extern __shared__ float shared2[];
  float* const shared_val = shared2;
  float* const shared_grad = shared2 + 4 * blockDim.x;
  float* const shared_momentum = shared2 + 8 * blockDim.x;
  const uint shared_offset[stages_count] = {0, 16 * blockDim.x};  // Fix: Byte offsets
  const float beta1_f = adam_p.beta1;
  const float beta2_f = adam_p.beta2;
  const float epsilon_f = adam_p.epsilon;
  const float max_grad_1 = general_p.max_grad_1;
  unsigned int curr_stage_store = 0;
  unsigned int curr_stage_compute = 0;
  float lower_lr_bound = 0.0f;
  float upper_lr_bound = std::numeric_limits<float>::max();
  const float inv_n = 1.0f / static_cast<float>(num_gaussians);
  uint32_t this_step = 0;
  float this_lr_scale = 1.0f;
#define PREFETCH_SHM(type, field_name) \
  pipeline.producer_acquire(); \
  { \
    const uint cpy_bytes = ((sizeof(type) * this_block_range + 15) / 16) * 16; /* Align to 16B */\
    const uint cpy_bytes2 = ((sizeof(type) * this_block_range * 2 + 15) / 16) * 16; \
    cuda::memcpy_async(block, shared_val + shared_offset[curr_stage_store], \
                       reinterpret_cast<const float*>((field_name) + block_leader_thread_idx), \
                       cuda::aligned_size_t<16>(cpy_bytes), pipeline); \
    cuda::memcpy_async(block, shared_grad + shared_offset[curr_stage_store], \
                       reinterpret_cast<const float*>(field_name##_grad + block_leader_thread_idx), \
                       cuda::aligned_size_t<16>(cpy_bytes), pipeline); \
    cuda::memcpy_async(block, shared_momentum + shared_offset[curr_stage_store], \
                       reinterpret_cast<const float*>(field_name##_first_second + 2 * block_leader_thread_idx), \
                       cuda::aligned_size_t<16>(cpy_bytes2), pipeline); \
  } \
  pipeline.producer_commit(); \
  curr_stage_store = (curr_stage_store + 1) % stages_count
  PREFETCH_SHM(vec3, means);
  PREFETCH_SHM(float, opacities);

#define apply_(v, g, f, s, lr) adam_step_func((v), (g), (f), (s), (lr), beta1_f, beta2_f, epsilon_f, max_grad_1, lower_lr_bound, upper_lr_bound)
  // Means
  {
    pipeline.consumer_wait();
    if (enable) {
      vec3* pval = reinterpret_cast<vec3*>(shared_val + shared_offset[curr_stage_compute]);
      const vec3* pgrad = reinterpret_cast<const vec3*>(shared_grad + shared_offset[curr_stage_compute]);
      vec3* pmom = reinterpret_cast<vec3*>(shared_momentum + shared_offset[curr_stage_compute]);
      vec3 val = pval[local_idx];
      vec3 grad = pgrad[local_idx] * gradient_scale;  // Fix: Add gradient_scale
      vec3 first = pmom[local_idx * 2];
      vec3 second = pmom[local_idx * 2 + 1];
      float grad_norm_1 = sum(abs(grad));
      if (general_p.skip_zero_grad && grad_norm_1 == 0.0f) {  // Fix: Conditional skip, only on means
        enable = false;
      } else {
        this_step = ++gaussian_steps[idx];
        // Fix: Compute AdaBound bounds per gaussian (based on this_step)
        if (adam_p.enable_adabound) {
          // Use float intermediates and guard denominators to avoid underflow to zero.
          const float denom = fmaxf((1.0f - adam_p.beta2) * (float)this_step + 1.0f, 1e-20f);
          const float lower = 0.1f - 0.1f / denom;
          const float denom2 = fmaxf((1.0f - adam_p.beta2) * (float)this_step, 1e-20f);
          const float upper = 0.1f + 0.1f / denom2;
          lower_lr_bound = fmaxf(lower, 0.0f);
          // ensure upper bound is not smaller than lower bound (tiny epsilon)
          upper_lr_bound = fmaxf(upper, lower_lr_bound + 1e-12f);
        }
        if (this_step < 4096) {
          // Compute in float precision and guard the denominator to avoid numerical issues
          const float b2t = powf(adam_p.beta2, (float)this_step);
          const float b1t = powf(adam_p.beta1, (float)this_step);
          const float num = fmaxf(1.0f - b2t, 1e-30f);
          float den = 1.0f - b1t;
          den = fmaxf(den, 1e-16f);
          this_lr_scale = sqrtf(num) / den;
        }
        const float lr = general_p.means_lr * global_lr * this_lr_scale;
        apply_(val.x, grad.x, first.x, second.x, lr);
        apply_(val.y, grad.y, first.y, second.y, lr);
        apply_(val.z, grad.z, first.z, second.z, lr);
        means[idx] = val;
        means_first_second[2 * idx] = first;
        means_first_second[2 * idx + 1] = second;
      }
    }
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
    PREFETCH_SHM(vec4, rotations);
  }
  // Opacities
  {
    pipeline.consumer_wait();
    if (enable) {
      float* pval = reinterpret_cast<float*>(shared_val + shared_offset[curr_stage_compute]);
      const float* pgrad = reinterpret_cast<const float*>(shared_grad + shared_offset[curr_stage_compute]);
      float* pmom = reinterpret_cast<float*>(shared_momentum + shared_offset[curr_stage_compute]);
      float val = pval[local_idx];
      float grad = pgrad[local_idx] * gradient_scale + general_p.opacities_l1 * activate_opacity_deriv(val) * inv_n;
      float first = pmom[local_idx * 2];
      float second = pmom[local_idx * 2 + 1];
      const float lr = general_p.opacities_lr * global_lr * this_lr_scale;
      apply_(val, grad, first, second, lr);
      opacities[idx] = val;
      opacities_first_second[2 * idx] = first;
      opacities_first_second[2 * idx + 1] = second;
    }
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
    PREFETCH_SHM(vec3, scales);
  }
  // Rotations
  {
    pipeline.consumer_wait();
    if (enable) {
      vec4* pval = reinterpret_cast<vec4*>(shared_val + shared_offset[curr_stage_compute]);
      const vec4* pgrad = reinterpret_cast<const vec4*>(shared_grad + shared_offset[curr_stage_compute]);
      vec4* pmom = reinterpret_cast<vec4*>(shared_momentum + shared_offset[curr_stage_compute]);
      vec4 val = pval[local_idx];
      vec4 grad = pgrad[local_idx] * gradient_scale;  // Fix: Add gradient_scale
      vec4 first = pmom[local_idx * 2];
      vec4 second = pmom[local_idx * 2 + 1];
      const float lr = general_p.rotations_lr * global_lr * this_lr_scale;
      apply_(val.x, grad.x, first.x, second.x, lr);
      apply_(val.y, grad.y, first.y, second.y, lr);
      apply_(val.z, grad.z, first.z, second.z, lr);
      apply_(val.w, grad.w, first.w, second.w, lr);
      rotations[idx] = val;
      rotations_first_second[2 * idx] = first;
      rotations_first_second[2 * idx + 1] = second;
    }
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
  }
  // Scales
  {
    pipeline.consumer_wait();
    if (enable) {
      vec3* pval = reinterpret_cast<vec3*>(shared_val + shared_offset[curr_stage_compute]);
      const vec3* pgrad = reinterpret_cast<const vec3*>(shared_grad + shared_offset[curr_stage_compute]);
      vec3* pmom = reinterpret_cast<vec3*>(shared_momentum + shared_offset[curr_stage_compute]);
      vec3 val = pval[local_idx];
      vec3 grad = pgrad[local_idx] * gradient_scale +  // Fix: Add gradient_scale and L1 reg
                  general_p.scales_l1 * activate_scale_deriv(val) * inv_n;
      vec3 first = pmom[local_idx * 2];
      vec3 second = pmom[local_idx * 2 + 1];
      const float lr = general_p.scales_lr * global_lr * this_lr_scale;
      apply_(val.x, grad.x, first.x, second.x, lr);
      apply_(val.y, grad.y, first.y, second.y, lr);
      apply_(val.z, grad.z, first.z, second.z, lr);
      scales[idx] = val;
      scales_first_second[2 * idx] = first;
      scales_first_second[2 * idx + 1] = second;
    }
    curr_stage_compute = (curr_stage_compute + 1) % stages_count;
    pipeline.consumer_release();
  }
}

// Per-element adam step for SoA SH buffers with interleaved first/second moments.
// The gaussian_steps lookup uses gaussian_idx = element_idx % N (SoA stride).
__global__ static void adamw_sh_soa_step(
    float* __restrict__ thetas,
    const float* __restrict__ thetas_grad,
    float* __restrict__ first_second,
    const uint32_t* __restrict__ gaussian_steps,
    int num_elements,  // total SoA floats (num_coeffs * 3 * N)
    int N,             // gaussian count (SoA stride)
    AdamWParameters adam_p,
    GaussianOptimizationParams general_p,
    float lr,
    float gradient_scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    int gaussian_idx = tid % N;
    uint32_t this_step = __ldg(gaussian_steps + gaussian_idx);
    if (this_step == 0) return;  // skip if gaussian hasn't been stepped

    float this_lr_scale = 1.0f;
    if (this_step < 4096) {
        float b2t = powf(adam_p.beta2, (float)this_step);
        float b1t = powf(adam_p.beta1, (float)this_step);
        float num = fmaxf(1.0f - b2t, adam_p.epsilon);
        float den = fmaxf(1.0f - b1t, adam_p.epsilon);
        this_lr_scale = sqrtf(num) / den;
    }

    float lower_lr_bound = 0.0f;
    float upper_lr_bound = std::numeric_limits<float>::max();
    if (adam_p.enable_adabound) {
        float denom = fmaxf((1.0f - adam_p.beta2) * (float)this_step + 1.0f, 1e-20f);
        float lower = 0.1f - 0.1f / denom;
        float denom2 = fmaxf((1.0f - adam_p.beta2) * (float)this_step, 1e-20f);
        float upper = 0.1f + 0.1f / denom2;
        lower_lr_bound = fmaxf(lower, 0.0f);
        upper_lr_bound = fmaxf(upper, lower_lr_bound + 1e-12f);
    }

    float val = thetas[tid];
    float grad = thetas_grad[tid] * gradient_scale;
    float first = first_second[2 * tid];
    float second = first_second[2 * tid + 1];

    adam_step_func(val, grad, first, second, lr * this_lr_scale,
                   adam_p.beta1, adam_p.beta2, adam_p.epsilon, general_p.max_grad_1,
                   lower_lr_bound, upper_lr_bound);

    thetas[tid] = val;
    first_second[2 * tid] = first;
    first_second[2 * tid + 1] = second;
}

// Gathers SoA interleaved first_second data based on mapping.
__global__ static void gather_soa_interleaved(
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
    int src_pos = channel * src_stride + mapping[item_idx];
    int dst_pos = channel * num_items + item_idx;
    dst[2 * dst_pos] = src[2 * src_pos];
    dst[2 * dst_pos + 1] = src[2 * src_pos + 1];
}

static void gather_soa_interleaved_buffers(
    const thrust::device_vector<float>& src,
    thrust::device_vector<float>& dst,
    const uint* mapping,
    int num_items, int num_channels, int src_stride
) {
    int total = num_items * num_channels;
    dst.resize(total * 2, 0.f);
    int grid = (total + 255) / 256;
    gather_soa_interleaved<<<grid, 256>>>(
        thrust::raw_pointer_cast(src.data()),
        thrust::raw_pointer_cast(dst.data()),
        mapping, num_items, num_channels, src_stride);
}

static void relayout_soa_interleaved(
    thrust::device_vector<float>& buf,
    int old_n, int new_n, int num_channels
) {
    if (old_n == 0) { buf.assign(new_n * num_channels * 2, 0.f); return; }
    thrust::device_vector<uint> identity(old_n);
    thrust::sequence(thrust::device, identity.begin(), identity.end());
    thrust::device_vector<float> new_buf(new_n * num_channels * 2, 0.f);
    int total = old_n * num_channels;
    int grid = (total + 255) / 256;
    gather_soa_interleaved<<<grid, 256>>>(
        thrust::raw_pointer_cast(buf.data()),
        thrust::raw_pointer_cast(new_buf.data()),
        thrust::raw_pointer_cast(identity.data()),
        old_n, num_channels, old_n);
    buf = std::move(new_buf);
}

struct adamw_domain {
  static constexpr char const *name{"fast_gs"};
};
using range = nvtx3::scoped_range_in<adamw_domain>;
using attr = nvtx3::event_attributes;
using regstr = nvtx3::registered_string_in<adamw_domain>;
using ncat = nvtx3::named_category_in<adamw_domain>;
struct m_gs_major {
  static constexpr char const *message{"adam_major"};
};
struct m_sh_rest {
  static constexpr char const *message{"adam_sh_rest"};
};

void AdamW::step(float scale, cudaStream_t stream) {

  NVTX3_FUNC_RANGE();
  const float gradient_scale = scale;  // This is the gradient scaler, not learning rate multiplier
  constexpr int block_size = 256;
  auto n = m_gaussians->size();
  const int grid = (n + block_size - 1) / block_size;

  if (!m_gaussians || !m_gaussians_grad) {
    throw std::runtime_error("AdamW::step: gaussians or gaussians_grad is null");
  } else if (m_gaussians->size() != m_gaussians_grad->size()) {
    throw std::runtime_error("AdamW::step: gaussians and gaussians_grad must have same size");
  }

  auto expected_shm = 4 * block_size * sizeof(float) * 4 * 2;
  {
    auto msg = regstr::get<m_gs_major>();
    nvtx3::event_attributes attr(msg, nvtx3::payload{n});
    range range(attr);

    launch_gaussian_adam_step_SoA_ref2<<<grid, block_size, expected_shm, stream>>>(
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
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      m_adam_params,
      m_params,
      m_gaussians->size(),
      gradient_scale,
      m_global_lr
    );
    maybe_sync(stream);
  }

  // SH0 - degree 0 (1 coeff, 3*N elements), lr = shs_lr
  {
    int grid_sh0 = (3 * n + 255) / 256;
    adamw_sh_soa_step<<<grid_sh0, 256, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh0().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh0().data()),
      thrust::raw_pointer_cast(m_sh0_first_second.data()),
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      3 * n, n, m_adam_params, m_params, m_params.shs_lr * m_global_lr, gradient_scale);
    maybe_sync(stream);
  }

  // SH1 - degree 1 (3 coeffs, 9*N elements), lr = shs_lr * 0.05
  {
    int grid_sh1 = (9 * n + 255) / 256;
    adamw_sh_soa_step<<<grid_sh1, 256, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh1().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh1().data()),
      thrust::raw_pointer_cast(m_sh1_first_second.data()),
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      9 * n, n, m_adam_params, m_params, m_params.shs_lr * 0.05f * m_global_lr, gradient_scale);
    maybe_sync(stream);
  }

  // SH2 - degree 2 (5 coeffs, 15*N elements), lr = shs_lr * 0.05
  {
    int grid_sh2 = (15 * n + 255) / 256;
    adamw_sh_soa_step<<<grid_sh2, 256, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh2().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh2().data()),
      thrust::raw_pointer_cast(m_sh2_first_second.data()),
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      15 * n, n, m_adam_params, m_params, m_params.shs_lr * 0.05f * m_global_lr, gradient_scale);
    maybe_sync(stream);
  }

  // SH3 - degree 3 (7 coeffs, 21*N elements), lr = shs_lr * 0.05
  {
    int grid_sh3 = (21 * n + 255) / 256;
    adamw_sh_soa_step<<<grid_sh3, 256, 0, stream>>>(
      thrust::raw_pointer_cast(m_gaussians->sh3().data()),
      thrust::raw_pointer_cast(m_gaussians_grad->sh3().data()),
      thrust::raw_pointer_cast(m_sh3_first_second.data()),
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      21 * n, n, m_adam_params, m_params, m_params.shs_lr * 0.05f * m_global_lr, gradient_scale);
    maybe_sync(stream);
  }
}


AdamW::AdamW(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
  OptimizerBase(gaussians, gaussians_grad) {
  // Resize and reset all internal buffers
  AdamW::reset();
}

__global__ void copy_optimizer_base_state(
  const vec3 * __restrict__ src_means,
  vec3 * __restrict__ dst_means,
  const float * __restrict__ src_opacities,
  float * __restrict__ dst_opacities,
  const vec4 * __restrict__ src_rotations,
  vec4 * __restrict__ dst_rotations,
  const vec3 * __restrict__ src_scales,
  vec3 * __restrict__ dst_scales,
  const uint32_t* __restrict__ src_gaussian_steps,
  uint32_t* __restrict__ dst_gaussian_steps,
  const uint * __restrict__ mapping,
  int num_items
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  uint src_idx = mapping[idx];

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

  // Copy gaussian steps
  dst_gaussian_steps[idx] = src_gaussian_steps[src_idx];
}

void AdamW::remove(char* kept_flag, int num_kept) {
  // filters the gaussians' first second.
  size_t original_size = m_gaussians->size();
  thrust::device_vector<uint> mapping(original_size); // mapping[idx] = original_idx

  thrust::copy_if(
    thrust::device,
    thrust::make_counting_iterator<uint>(0), thrust::make_counting_iterator<uint>(original_size),
    mapping.begin(), [kept_flag] __device__ (uint orig) { return static_cast<bool>(kept_flag[orig]); });

  thrust::device_vector<vec3> means(2 * num_kept);
  thrust::device_vector<float> opacities(2 * num_kept);
  thrust::device_vector<vec4> rotations(2 * num_kept);
  thrust::device_vector<vec3> scales(2 * num_kept);
  thrust::device_vector<uint32_t> gaussian_steps(num_kept);

  const int grid = (num_kept + 255) / 256;
  copy_optimizer_base_state<<<grid, 256>>>(
      thrust::raw_pointer_cast(m_means_first_second.data()),
      thrust::raw_pointer_cast(means.data()),
      thrust::raw_pointer_cast(m_opacities_first_second.data()),
      thrust::raw_pointer_cast(opacities.data()),
      thrust::raw_pointer_cast(m_rotations_first_second.data()),
      thrust::raw_pointer_cast(rotations.data()),
      thrust::raw_pointer_cast(m_scales_first_second.data()),
      thrust::raw_pointer_cast(scales.data()),
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      thrust::raw_pointer_cast(gaussian_steps.data()),
      thrust::raw_pointer_cast(mapping.data()),
      num_kept
  );

  // Gather SH buffers using SoA interleaved layout
  thrust::device_vector<float> sh0, sh1, sh2, sh3;
  gather_soa_interleaved_buffers(m_sh0_first_second, sh0,
      thrust::raw_pointer_cast(mapping.data()), num_kept, 3, original_size);
  gather_soa_interleaved_buffers(m_sh1_first_second, sh1,
      thrust::raw_pointer_cast(mapping.data()), num_kept, 9, original_size);
  gather_soa_interleaved_buffers(m_sh2_first_second, sh2,
      thrust::raw_pointer_cast(mapping.data()), num_kept, 15, original_size);
  gather_soa_interleaved_buffers(m_sh3_first_second, sh3,
      thrust::raw_pointer_cast(mapping.data()), num_kept, 21, original_size);

  m_means_first_second = std::move(means);
  m_opacities_first_second = std::move(opacities);
  m_rotations_first_second = std::move(rotations);
  m_scales_first_second = std::move(scales);
  m_sh0_first_second = std::move(sh0);
  m_sh1_first_second = std::move(sh1);
  m_sh2_first_second = std::move(sh2);
  m_sh3_first_second = std::move(sh3);
  m_gaussian_steps = std::move(gaussian_steps);
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
  // Step counts
  uint32_t* __restrict__ gaussian_steps,
  // Indices
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

  // Copy step count: but with half the value to ensure it could be optimized efficiently.
  gaussian_steps[dst_idx] = (gaussian_steps[src_idx]);
}

void AdamW::duplicate(int* indices, int* new_indices, int num_duplicate) {
  if (num_duplicate == 0) return;
  int old_n = m_gaussians->size();
  int new_n = old_n + num_duplicate;

  // Resize vectors to accommodate duplicated gaussians
  m_means_first_second.resize(new_n * 2, vec3(0.f, 0.f, 0.f));
  m_opacities_first_second.resize(new_n * 2, 0.f);
  m_rotations_first_second.resize(new_n * 2, vec4(0.f, 0.f, 0.f, 0.f));
  m_scales_first_second.resize(new_n * 2, vec3(0.f, 0.f, 0.f));
  m_gaussian_steps.resize(new_n, 0);

  // Relayout SH buffers using SoA interleaved layout
  relayout_soa_interleaved(m_sh0_first_second, old_n, new_n, 3);
  relayout_soa_interleaved(m_sh1_first_second, old_n, new_n, 9);
  relayout_soa_interleaved(m_sh2_first_second, old_n, new_n, 15);
  relayout_soa_interleaved(m_sh3_first_second, old_n, new_n, 21);

  // TODO: This design does not provide better result. Why?
  // const int grid = (num_duplicate + 255) / 256;
  // duplicate_optimizer_state_kernel<<<grid, 256>>>(
  //   thrust::raw_pointer_cast(m_means_first_second.data()),
  //   thrust::raw_pointer_cast(m_opacities_first_second.data()),
  //   thrust::raw_pointer_cast(m_rotations_first_second.data()),
  //   thrust::raw_pointer_cast(m_scales_first_second.data()),
  //   thrust::raw_pointer_cast(m_gaussian_steps.data()),
  //   indices,
  //   new_indices,
  //   num_duplicate
  // );
}

void AdamW::reset() {
  size_t num_gaussians = m_gaussians->size();

  m_means_first_second.resize(num_gaussians * 2, vec3(0.f));
  m_opacities_first_second.resize(num_gaussians * 2, 0.f);
  m_rotations_first_second.resize(num_gaussians * 2, vec4(0.f));
  m_scales_first_second.resize(num_gaussians * 2, vec3(0.f));
  m_sh0_first_second.assign(num_gaussians * 3 * 2, 0.f);
  m_sh1_first_second.assign(num_gaussians * 9 * 2, 0.f);
  m_sh2_first_second.assign(num_gaussians * 15 * 2, 0.f);
  m_sh3_first_second.assign(num_gaussians * 21 * 2, 0.f);
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
      sh0_first_second = m_sh0_first_second.data(),
      sh1_first_second = m_sh1_first_second.data(),
      sh2_first_second = m_sh2_first_second.data(),
      sh3_first_second = m_sh3_first_second.data(),
      gaussian_steps = m_gaussian_steps.data(),
      N = static_cast<int>(num_gaussians)
    ] __device__(int idx) {
      means_first_second[idx * 2] = vec3(0.f);
      means_first_second[idx * 2 + 1] = vec3(0.f);
      opacities_first_second[idx * 2] = 0.f;
      opacities_first_second[idx * 2 + 1] = 0.f;
      rotations_first_second[idx * 2] = vec4(0.f);
      rotations_first_second[idx * 2 + 1] = vec4(0.f);
      scales_first_second[idx * 2] = vec3(0.f);
      scales_first_second[idx * 2 + 1] = vec3(0.f);
      // Zero SH0 (3 channels)
      for (int c = 0; c < 3; c++) {
        int pos = c * N + idx;
        sh0_first_second[2 * pos] = 0.f;
        sh0_first_second[2 * pos + 1] = 0.f;
      }
      // Zero SH1 (9 channels)
      for (int c = 0; c < 9; c++) {
        int pos = c * N + idx;
        sh1_first_second[2 * pos] = 0.f;
        sh1_first_second[2 * pos + 1] = 0.f;
      }
      // Zero SH2 (15 channels)
      for (int c = 0; c < 15; c++) {
        int pos = c * N + idx;
        sh2_first_second[2 * pos] = 0.f;
        sh2_first_second[2 * pos + 1] = 0.f;
      }
      // Zero SH3 (21 channels)
      for (int c = 0; c < 21; c++) {
        int pos = c * N + idx;
        sh3_first_second[2 * pos] = 0.f;
        sh3_first_second[2 * pos + 1] = 0.f;
      }
      gaussian_steps[idx] = 0;
    }
  );
}

/// @brief Reorder Gaussians based on provided indices
void AdamW::reorder(uint* indices) {
  int num_gaussians = m_means_first_second.size() / 2;
  // Create temporary vectors for reordered data
  thrust::device_vector<vec3> means(num_gaussians * 2);
  thrust::device_vector<float> opacities(num_gaussians * 2);
  thrust::device_vector<vec4> rotations(num_gaussians * 2);
  thrust::device_vector<vec3> scales(num_gaussians * 2);
  thrust::device_vector<uint32_t> gaussian_steps(num_gaussians);

  const int grid = (num_gaussians + 255) / 256;
  copy_optimizer_base_state<<<grid, 256>>>(
      thrust::raw_pointer_cast(m_means_first_second.data()),
      thrust::raw_pointer_cast(means.data()),
      thrust::raw_pointer_cast(m_opacities_first_second.data()),
      thrust::raw_pointer_cast(opacities.data()),
      thrust::raw_pointer_cast(m_rotations_first_second.data()),
      thrust::raw_pointer_cast(rotations.data()),
      thrust::raw_pointer_cast(m_scales_first_second.data()),
      thrust::raw_pointer_cast(scales.data()),
      thrust::raw_pointer_cast(m_gaussian_steps.data()),
      thrust::raw_pointer_cast(gaussian_steps.data()),
      indices,
      num_gaussians
  );

  // Gather SH buffers using SoA interleaved layout
  thrust::device_vector<float> sh0, sh1, sh2, sh3;
  gather_soa_interleaved_buffers(m_sh0_first_second, sh0,
      indices, num_gaussians, 3, num_gaussians);
  gather_soa_interleaved_buffers(m_sh1_first_second, sh1,
      indices, num_gaussians, 9, num_gaussians);
  gather_soa_interleaved_buffers(m_sh2_first_second, sh2,
      indices, num_gaussians, 15, num_gaussians);
  gather_soa_interleaved_buffers(m_sh3_first_second, sh3,
      indices, num_gaussians, 21, num_gaussians);

  // Move reordered data back to member variables
  m_means_first_second = std::move(means);
  m_opacities_first_second = std::move(opacities);
  m_rotations_first_second = std::move(rotations);
  m_scales_first_second = std::move(scales);
  m_sh0_first_second = std::move(sh0);
  m_sh1_first_second = std::move(sh1);
  m_sh2_first_second = std::move(sh2);
  m_sh3_first_second = std::move(sh3);
  m_gaussian_steps = std::move(gaussian_steps);
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