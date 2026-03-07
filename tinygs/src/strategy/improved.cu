#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/platform/buffer_utils.hpp"
#include "random/device.cuh"
#include "tinygs/strategy/improved.hpp"
#include "tinygs/core/gaussian.hpp"
#include "tinygs/random/multinomial.hpp"
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

// Simple helper mirroring DefaultStrategy's reset of opacity
static void reset_opacity(const std::shared_ptr<GPUGaussian3d>& gaussians, float min_opacity_threshold, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(stream);

  thrust::for_each(
      exec,
      gaussians->opacities().begin(),
      gaussians->opacities().end(),
      [min_opacity_threshold] __device__(float& opacity) {
        opacity = deactivate_opacity(fminf(activate_opacity(opacity), min_opacity_threshold));
      });
}

ImprovedStrategy::ImprovedStrategy(
    std::shared_ptr<BackendRuntime> runtime,
    std::shared_ptr<GPUGaussian3d> gaussians,
    std::shared_ptr<GPUGaussian3d> gaussians_grad,
    std::shared_ptr<OptimizerBase> optimizer)
    : StrategyBase(runtime, gaussians, gaussians_grad, optimizer) {}

ImprovedStrategy::~ImprovedStrategy() = default;

__global__ static void add_noise_opacity(uint N, float noise_scale,
                                         float *__restrict__ opacities,
                                         const float *__restrict__ noise) {
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float opacity = activate_opacity(opacities[i]);
  // Low opacity => high noise, High opacity => low noise
  const float actual_noise = (1.01f - opacity) * noise_scale * noise[i];
  opacities[i] += actual_noise;
}

void ImprovedStrategy::step_impl(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue)));

  if (!ctx.densification_info) {
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(ctx.runtime, num_gaussians);
    fill_buffer_zero_async(ctx.runtime, ctx.queue, ctx.densification_info);
  }

  const int step = this_step();
  if (step % m_params.refine_every == 0 && step >= m_params.start_refine && step <= m_params.end_refine) {
    if (m_gaussians->size() < m_params.max_num_gaussians) {
      const float rate = static_cast<float>(this_step()) / m_params.end_refine;
      const int budget = min(int(sqrt(rate + 1e-10) * m_params.max_num_gaussians),
                             m_params.max_num_gaussians);
      duplicate(ctx, budget);
    }
    prune(ctx);

    // Reset densification info since indices may have changed.
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(ctx.runtime, num_gaussians);
    fill_buffer_zero_async(ctx.runtime, ctx.queue, ctx.densification_info);
  }

  if (m_params.reset_every > 0 && step % m_params.reset_every == 0 &&
      step >= m_params.start_refine && step < m_params.end_refine) {
    // this scale is larger than default (10 vs. 2)
    reset_opacity(m_gaussians, 2.f * m_params.pruning_opacity_threshold, to_cuda_stream(ctx.queue));
    on_reset_opacity(ctx.queue);
  }

  if (m_noise_lr_init > 0) {
    auto N = m_gaussians->size();
    float noise_scale = m_noise_lr_init * m_optimizer->get_lr() *
                        m_optimizer->get_optimization_params().opacities_lr;
    thrust::device_vector<float> noise(N);
    generate_random_logistic(m_rng, N, thrust::raw_pointer_cast(noise.data()));
    linear_kernel(add_noise_opacity, 0, to_cuda_stream(ctx.queue), N, noise_scale,
                  thrust::raw_pointer_cast(m_gaussians->opacities().data()),
                  thrust::raw_pointer_cast(noise.data()));
  }
}

void ImprovedStrategy::reset() {
  // No persistent state to clear.
}

void ImprovedStrategy::duplicate(const RasterizeContext& ctx, int budget) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
  const int num_gaussians = static_cast<int>(m_gaussians->size());

  if (!ctx.densification_info || static_cast<int>(buffer_count<DensificationInfo>(ctx.densification_info)) != num_gaussians) {
    log_warning("Densification info is not provided or has wrong size, skip duplication.");
    return;
  }

  auto* d_densification_info = buffer_data<DensificationInfo>(ctx.densification_info);

  // Compute gradient value per gaussian
  thrust::device_vector<float> grad_values(num_gaussians, 0.f);
  auto* d_grad_values = thrust::raw_pointer_cast(grad_values.data());

  thrust::for_each(                                       //
      exec,                                               //
      thrust::make_counting_iterator<int>(0),             //
      thrust::make_counting_iterator<int>(num_gaussians), //
      [d_densification_info, d_grad_values] __device__(int i) {
        const float accum = d_densification_info[i].accum_absgrad_mean2d; // it must use absgrad
        const float denom = fmaxf(d_densification_info[i].accum_counter, 1.0f);
        d_grad_values[i] = accum / denom;
      });

  const float grow_grad = m_params.duplicate_grad_threshold * ctx.grad_scaler;

  // Flag candidates exceeding threshold
  thrust::device_vector<char> candidate_flags(num_gaussians, 0);
  auto* d_candidate_flags = thrust::raw_pointer_cast(candidate_flags.data());
  thrust::for_each(exec,                                  //
                   thrust::make_counting_iterator<int>(0),
                   thrust::make_counting_iterator<int>(num_gaussians),
                   [d_candidate_flags, d_grad_values, grow_grad] __device__(int i) {
                     d_candidate_flags[i] = d_grad_values[i] >= grow_grad ? 1 : 0;
                   });

  const int num_candidates = thrust::transform_reduce(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      [d_candidate_flags] __host__ __device__(int i) {
        return d_candidate_flags[i] != 0 ? 1 : 0;
      },
      0, thrust::plus<int>());

  if (num_candidates <= 0) {
    return;
  }

  // We add one child per selected gaussian (split along longest axis)
  const int max_new = budget - num_gaussians;
  const int num_grows = max(0, min(num_candidates, max_new));
  if (num_grows <= 0) {
    return;
  }

  // Sample candidate indices via multinomial (with replacement) using grad-based weights
  thrust::device_vector<float> grow_weights(num_gaussians, 0.f);
  auto* d_grow_weights = thrust::raw_pointer_cast(grow_weights.data());
  thrust::for_each(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      [d_candidate_flags, d_grad_values, d_grow_weights] __device__(int i) {
        // Use gradient magnitude as sampling weight for candidates; zero otherwise
        d_grow_weights[i] = d_candidate_flags[i] != 0 ? fmaxf(d_grad_values[i], 1e-20f) : 0.0f;
      });

  // Collect candidate indices (we take the first num_grows to avoid heavy sorts)
  thrust::device_vector<int> grow_indices_src(num_grows);
  std::shared_ptr<BackendBuffer> grow_indices_src_sampled_buf;
  int *d_grow_indices_src = thrust::raw_pointer_cast(grow_indices_src.data());

  if (num_grows == num_candidates) {
    auto* out = thrust::copy_if(exec, thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians), d_candidate_flags,
    d_grow_indices_src, [] __device__(char f) { return f != 0; });
    const int copied = static_cast<int>(out - d_grow_indices_src);
    if (copied < num_grows) {
      log_warning("Grow source indices fewer than expected: {} < {}", copied, num_grows);
    }
  } else {
    grow_indices_src_sampled_buf = multinomial_cuda_with_replacement(
        d_grow_weights,                                // weights on device
        num_gaussians,                                 // categories
        num_grows,                                     // samples to draw
        static_cast<int>(m_params.seed + this_step()), // seed varies with step
      ctx.queue.get());
    d_grow_indices_src = buffer_data<int>(grow_indices_src_sampled_buf);
  }

  // Allocate target indices (appended range)
  thrust::device_vector<int> grow_indices_target(num_grows);
  auto* d_grow_indices_target = thrust::raw_pointer_cast(grow_indices_target.data());
  thrust::copy(exec, thrust::make_counting_iterator<int>(num_gaussians),
               thrust::make_counting_iterator<int>(num_gaussians + num_grows), d_grow_indices_target);

  // Append and update optimizer state
  StrategyBase::on_duplicate(d_grow_indices_src, d_grow_indices_target, num_grows, ctx.queue);
  if (static_cast<int>(m_gaussians->size()) != num_gaussians + num_grows) {
    log_error("Grow gaussians failed, expected {} gaussians, but got {}", num_gaussians + num_grows,
              m_gaussians->size());
  }

  // Perform long-axis split updates
  const float rate = m_split_distance > 0.f ? m_split_distance : m_params.duplicate_scale_threshold;
  const float reduction = m_opacity_reduction;

  thrust::for_each(exec, thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(num_grows),
      [d_grow_indices_src, d_grow_indices_target,                                 //
      means3d = thrust::raw_pointer_cast(m_gaussians->means().data()),           //
      scales3d = thrust::raw_pointer_cast(m_gaussians->scales().data()),         //
      opacities = thrust::raw_pointer_cast(m_gaussians->opacities().data()),     //
      rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data()),     //
      sh0_data = thrust::raw_pointer_cast(m_gaussians->sh0().data()),            //
      sh1_data = thrust::raw_pointer_cast(m_gaussians->sh1().data()),            //
      sh2_data = thrust::raw_pointer_cast(m_gaussians->sh2().data()),            //
      sh3_data = thrust::raw_pointer_cast(m_gaussians->sh3().data()),            //
      N = static_cast<int>(m_gaussians->size()),                                 //
      rate, reduction] __device__(int i) {
        const int src_idx = d_grow_indices_src[i];
        const int target_idx = d_grow_indices_target[i];

        // Copy rotation and SHs (SoA per-degree layout)
        rotations[target_idx] = rotations[src_idx];
        // SH0: 1 coeff * 3 channels = 3 entries
        for (int ch = 0; ch < 3; ch++) {
          sh0_data[ch * N + target_idx] = sh0_data[ch * N + src_idx];
        }
        // SH1: 3 coeffs * 3 channels = 9 entries
        for (int ch = 0; ch < 9; ch++) {
          sh1_data[ch * N + target_idx] = sh1_data[ch * N + src_idx];
        }
        // SH2: 5 coeffs * 3 channels = 15 entries
        for (int ch = 0; ch < 15; ch++) {
          sh2_data[ch * N + target_idx] = sh2_data[ch * N + src_idx];
        }
        // SH3: 7 coeffs * 3 channels = 21 entries
        for (int ch = 0; ch < 21; ch++) {
          sh3_data[ch * N + target_idx] = sh3_data[ch * N + src_idx];
        }

        // Rotation matrix from quaternion (w, x, y, z) stored as (x,y,z,w)
        float r = rotations[src_idx].x;
        float x = rotations[src_idx].y;
        float y = rotations[src_idx].z;
        float z = rotations[src_idx].w;
        // Normalize quaternion to unit length
        const float quat_mag = sqrtf(r * r + x * x + y * y + z * z);
        if (quat_mag > 1e-8f) {
          const float inv_mag = 1.0f / quat_mag;
          r *= inv_mag;
          x *= inv_mag;
          y *= inv_mag;
          z *= inv_mag;
        }
        glm::mat3 rot = glm::mat3(
          1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
          2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
          2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

        // Longest axis of activated scale
        const vec3 actual_scale = activate_scale(scales3d[src_idx]);
        int max_axis = 0;
        float max_val = actual_scale.x;
        if (actual_scale.y > max_val) {
          max_axis = 1;
          max_val = actual_scale.y;
        }
        if (actual_scale.z > max_val) {
          max_axis = 2;
          max_val = actual_scale.z;
        }

        // Offset along longest axis in local space and rotate to world
        vec3 dir = vec3(0.f, 0.f, 0.f);
        if (max_axis == 0)
          dir.x = 1.f;
        else if (max_axis == 1)
          dir.y = 1.f;
        else
          dir.z = 1.f;

        const vec3 off_local = dir * (max_val * rate);
        const vec3 off_world = rot * off_local;

        // Place new gaussian and move source opposite direction
        means3d[target_idx] = means3d[src_idx] + off_world;
        means3d[src_idx] = means3d[src_idx] - off_world;

        // Scale update: rate_w/rate_h on longest axis, overall rate_h
        const float rate_w = 1.f - rate;
        const float rate_h = sqrtf(fmaxf(0.f, 1.f - rate * rate));
        vec3 new_scale = actual_scale;
        if (max_axis == 0)
          new_scale.x = max_val * rate_w / rate_h;
        else if (max_axis == 1)
          new_scale.y = max_val * rate_w / rate_h;
        else
          new_scale.z = max_val * rate_w / rate_h;
        new_scale = new_scale * rate_h;

        scales3d[target_idx] = deactivate_scale(new_scale);
        scales3d[src_idx] = deactivate_scale(new_scale);

        // Opacity reduction for both children
        const float old_opacity = activate_opacity(opacities[src_idx]);
        const float new_opacity = old_opacity * reduction;
        opacities[target_idx] = deactivate_opacity(new_opacity);
        opacities[src_idx] = deactivate_opacity(new_opacity);
      });

  log_info("Add {} gaussians via long-axis split ({} -> {})", num_grows, num_gaussians, m_gaussians->size());
}

void ImprovedStrategy::prune(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));

  const int num_gaussians = static_cast<int>(m_gaussians->size());
  thrust::device_vector<char> is_alive(num_gaussians);

  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  thrust::for_each(exec,                                                     //
                   thrust::make_counting_iterator<int>(0),                   //
                   thrust::make_counting_iterator<int>(num_gaussians),      //
                   [d_is_alive = is_alive.data(), d_opacity,                //
                    scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),       //
                    scene_scale = m_gaussians->scene_scale(),                             //
                    rotation = thrust::raw_pointer_cast(m_gaussians->rotations().data()), //
                    pruning_scale_threshold = m_params.pruning_scale_threshold,           //
                    prune_large = this_step() > m_params.reset_every,                     //
                    min_opacity = m_params.pruning_opacity_threshold] __device__(int i) {
                     bool not_large_ws = max(activate_scale(scale[i])) < pruning_scale_threshold * scene_scale;
                     bool not_transparent = activate_opacity(d_opacity[i]) > min_opacity;
                     bool not_degenerate = sum(abs(rotation[i])) > FLT_EPSILON;

                     if (not_transparent && (not_large_ws || !prune_large) && not_degenerate) {
                       d_is_alive[i] = 1;
                     } else {
                       d_is_alive[i] = 0;
                     }
                   });

  int nums_kept = thrust::transform_reduce(exec, thrust::make_counting_iterator<int>(0),
                                           thrust::make_counting_iterator<int>(num_gaussians),
                                           [ia = is_alive.data()] __device__(int i) -> int { return ia[i] != 0 ? 1 : 0; },
                                           0, thrust::plus<int>());

  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept, ctx.queue);
  log_info("Remove {} dead gaussians (kept {})", num_gaussians - nums_kept, nums_kept);
}

void ImprovedStrategy::set_params(const json& config) {
  StrategyBase::set_params(config);
  m_rng.seed(m_params.seed);
  if (config.contains("split_distance")) {
    m_split_distance = config["split_distance"].get<float>();
  }
  if (config.contains("opacity_reduction")) {
    m_opacity_reduction = config["opacity_reduction"].get<float>();
  }
  if (config.contains("noise_lr_init")) {
    m_noise_lr_init = config["noise_lr_init"].get<float>();
  }
}

json ImprovedStrategy::get_params() const {
  json params = StrategyBase::get_params();
  params["type"] = "improved";
  params["split_distance"] = m_split_distance;
  params["opacity_reduction"] = m_opacity_reduction;
  params["noise_lr_init"] = m_noise_lr_init;
  return params;
}

} // namespace tinygs
