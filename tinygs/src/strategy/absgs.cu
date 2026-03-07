// AbsGS densification strategy.
// Clone uses standard gradients; split uses absolute gradients with a separate threshold.
// Reference: ref_impl/AbsGS/scene/gaussian_model.py

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <nvtx3/nvtx3.hpp>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/platform/buffer_utils.hpp"
#include "random/device.cuh"
#include "tinygs/strategy/absgs.hpp"

namespace tinygs {

AbsGSStrategy::AbsGSStrategy(
    std::shared_ptr<BackendRuntime> runtime,
    std::shared_ptr<GPUGaussian3d> gaussians,
    std::shared_ptr<GPUGaussian3d> gaussians_grad,
    std::shared_ptr<OptimizerBase> optimizer)
    : StrategyBase(runtime, gaussians, gaussians_grad, optimizer) {}

AbsGSStrategy::~AbsGSStrategy() = default;

static void reset_opacity_absgs(const std::shared_ptr<GPUGaussian3d>& gaussians,
                                float min_opacity_threshold, cudaStream_t stream) {
  auto exec = thrust::cuda::par.on(stream);
  thrust::for_each(exec,
      gaussians->opacities().begin(),
      gaussians->opacities().end(),
      [min_opacity_threshold] __device__(float& opacity) {
        opacity = deactivate_opacity(fminf(activate_opacity(opacity), min_opacity_threshold));
      });
}

void AbsGSStrategy::step_impl(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue)));

  if (!ctx.densification_info) {
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(*ctx.runtime, num_gaussians);
    fill_buffer_zero_async(*ctx.runtime, *ctx.queue, ctx.densification_info);
  }

  const int step = this_step();
  if (step % m_params.refine_every == 0 &&
      step >= m_params.start_refine &&
      step <= m_params.end_refine) {
    if (m_gaussians->size() < m_params.max_num_gaussians) {
      duplicate(ctx);
    }
    prune(ctx);
    // Reset densification info since indices have changed
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(*ctx.runtime, num_gaussians);
    fill_buffer_zero_async(*ctx.runtime, *ctx.queue, ctx.densification_info);
  }

  if (m_params.reset_every > 0 && step % m_params.reset_every == 0 &&
      step >= m_params.start_refine && step < m_params.end_refine) {
    reset_opacity_absgs(m_gaussians, 2.f * m_params.pruning_opacity_threshold, to_cuda_stream(ctx.queue));
    on_reset_opacity(ctx.queue);
  }
}

void AbsGSStrategy::reset() {
  // No persistent state to clear.
}

void AbsGSStrategy::duplicate(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
  auto num_gaussians = m_gaussians->size();

  if (!ctx.densification_info || buffer_count<DensificationInfo>(ctx.densification_info) != num_gaussians) {
    log_warning("Densification info is not provided or has wrong size, skip duplication.");
    return;
  }

  // AbsGS key difference: separate flags for clone and split
  // Clone: grad_mean2d / counter >= clone_thresh  AND  max_scale <= percent_dense * scene_scale
  // Split: absgrad_mean2d / counter >= absgrad_thresh  AND  max_scale > percent_dense * scene_scale
  thrust::device_vector<char> grow_flags(num_gaussians, 0);
  auto* d_grow_flags = thrust::raw_pointer_cast(grow_flags.data());
  auto* d_densification_info = buffer_data<DensificationInfo>(ctx.densification_info);
  auto* d_scale = thrust::raw_pointer_cast(m_gaussians->scales().data());

  constexpr int kClone = 1;
  constexpr int kSplit = 2;

  const float scene_scale = m_gaussians->scene_scale();
  const float clone_thresh = m_params.duplicate_grad_threshold * ctx.grad_scaler;
  const float split_thresh = m_absgrad_threshold * ctx.grad_scaler;
  const float scale_boundary = m_percent_dense * scene_scale;

  thrust::for_each(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      [d_densification_info, d_scale, d_grow_flags,
       clone_thresh, split_thresh, scale_boundary] __device__(int i) {
        const float counter = fmaxf(d_densification_info[i].accum_counter, 1.0f);
        const float grad = d_densification_info[i].accum_grad_mean2d / counter;
        const float absgrad = d_densification_info[i].accum_absgrad_mean2d / counter;
        const float max_scale = max(activate_scale(d_scale[i]));

        if (counter > 0) {
          if (grad >= clone_thresh && max_scale <= scale_boundary) {
            d_grow_flags[i] = kClone;  // small Gaussian, clone
          } else if (absgrad >= split_thresh && max_scale > scale_boundary) {
            d_grow_flags[i] = kSplit;  // large Gaussian, split along covariance
          }
        }
      });

  const int num_grows = thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] != 0 ? 1 : 0; },
      0, thrust::plus<int>());

  if (num_grows == 0) return;

  const int num_clones = thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] == kClone ? 1 : 0; },
      0, thrust::plus<int>());

  log_info("[AbsGS] Add {} gaussians ({} clone, {} split, {} total)",
           num_grows, num_clones, num_grows - num_clones, num_grows + num_gaussians);

  // Collect source indices
  thrust::device_vector<int> grow_indices_src(num_grows);
  auto* d_grow_indices_src = thrust::raw_pointer_cast(grow_indices_src.data());
  thrust::copy_if(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      d_grow_flags, d_grow_indices_src,
      [] __device__(char f) { return f != 0; });

  // Allocate target indices
  thrust::device_vector<int> grow_indices_target(num_grows);
  auto* d_grow_indices_target = thrust::raw_pointer_cast(grow_indices_target.data());
  thrust::copy(exec,
      thrust::make_counting_iterator<int>(num_gaussians),
      thrust::make_counting_iterator<int>(num_gaussians + num_grows),
      d_grow_indices_target);

  // Resize Gaussians + optimizer buffers
  StrategyBase::on_duplicate(d_grow_indices_src, d_grow_indices_target, num_grows, ctx.queue);

  // Generate random samples for split offsets
  thrust::device_vector<float> device_rng(num_grows * 6);
  generate_random_logistic(m_rng, num_grows * 6,
      thrust::raw_pointer_cast(device_rng.data()), 0.0f, 1.0f);

  // Apply clone or split to the new Gaussians
  const int n_new = static_cast<int>(m_gaussians->size());
  thrust::for_each(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_grows),
      [d_grow_indices_src, d_grow_indices_target, d_grow_flags,
       means3d = thrust::raw_pointer_cast(m_gaussians->means().data()),
       scales3d = thrust::raw_pointer_cast(m_gaussians->scales().data()),
       opacities = thrust::raw_pointer_cast(m_gaussians->opacities().data()),
       rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
       sh0_data = thrust::raw_pointer_cast(m_gaussians->sh0().data()),
       sh1_data = thrust::raw_pointer_cast(m_gaussians->sh1().data()),
       sh2_data = thrust::raw_pointer_cast(m_gaussians->sh2().data()),
       sh3_data = thrust::raw_pointer_cast(m_gaussians->sh3().data()),
       n_new,
       rng = thrust::raw_pointer_cast(device_rng.data())
      ] __device__(int i) {
        const int src_idx = d_grow_indices_src[i];
        const int target_idx = d_grow_indices_target[i];

        // Copy rotation and SH per-degree
        rotations[target_idx] = rotations[src_idx];
        for (int ch = 0; ch < 3; ch++)   sh0_data[ch * n_new + target_idx] = sh0_data[ch * n_new + src_idx];
        for (int ch = 0; ch < 9; ch++)   sh1_data[ch * n_new + target_idx] = sh1_data[ch * n_new + src_idx];
        for (int ch = 0; ch < 15; ch++)  sh2_data[ch * n_new + target_idx] = sh2_data[ch * n_new + src_idx];
        for (int ch = 0; ch < 21; ch++)  sh3_data[ch * n_new + target_idx] = sh3_data[ch * n_new + src_idx];

        if (d_grow_flags[src_idx] == kClone) {
          // Clone: exact duplicate
          means3d[target_idx] = means3d[src_idx];
          scales3d[target_idx] = scales3d[src_idx];
          opacities[target_idx] = opacities[src_idx];
        } else {
          // Split (AbsGS style): sample from Gaussian covariance, scale /= 1.6
          // Quaternion (w, x, y, z) stored as (x, y, z, w) in vec4
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

          const vec3 actual_scale = activate_scale(scales3d[src_idx]);
          const float new_opacity = 1.0f - sqrtf(1.0f - activate_opacity(opacities[src_idx]));
          const vec3 rand1 = vec3(rng[i * 6 + 0], rng[i * 6 + 1], rng[i * 6 + 2]);
          const vec3 rand2 = vec3(rng[i * 6 + 3], rng[i * 6 + 4], rng[i * 6 + 5]);
          const vec3 off1 = rot * (rand1 * (actual_scale + 1e-5f));
          const vec3 off2 = rot * (rand2 * (actual_scale + 1e-5f));

          // Target Gaussian
          means3d[target_idx] = means3d[src_idx] + off1;
          scales3d[target_idx] = deactivate_scale(actual_scale / 1.6f);
          opacities[target_idx] = deactivate_opacity(new_opacity);

          // Source Gaussian (also moved and shrunk)
          means3d[src_idx] = means3d[src_idx] + off2;
          scales3d[src_idx] = deactivate_scale(actual_scale / 1.6f);
          opacities[src_idx] = deactivate_opacity(new_opacity);
        }
      });
}

void AbsGSStrategy::prune(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
  const auto num_gaussians = m_gaussians->size();
  const int original_num_gaussians = buffer_count<DensificationInfo>(ctx.densification_info);
  const auto abs_ss_threshold = max(ctx.fwd_input.width, ctx.fwd_input.height) * m_params.max_screen_size;

  thrust::device_vector<char> is_alive(num_gaussians);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());

  thrust::for_each(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      [d_is_alive = is_alive.data(), d_opacity,
       scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),
       scene_scale = m_gaussians->scene_scale(),
       rotation = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
       pruning_scale_threshold = m_params.pruning_scale_threshold,
       prune_large = this_step() > m_params.reset_every,
       max_radii_threshold = abs_ss_threshold,
       original_num_gaussians,
       deninfo = buffer_data<DensificationInfo>(ctx.densification_info),
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) {
        bool not_large_ws = max(activate_scale(scale[i])) < pruning_scale_threshold * scene_scale;
        bool not_large_ss = i >= original_num_gaussians ||
                            deninfo[i].max_radii_screen < max_radii_threshold;
        bool not_transparent = activate_opacity(d_opacity[i]) > min_opacity;
        bool not_degenerate = sum(abs(rotation[i])) > FLT_EPSILON;

        if (not_transparent && ((not_large_ws && not_large_ss) || !prune_large) && not_degenerate) {
          d_is_alive[i] = 1;
        } else {
          d_is_alive[i] = 0;
        }
      });

  int nums_kept = thrust::transform_reduce(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(num_gaussians),
      [ia = is_alive.data()] __device__(int i) -> int { return ia[i] != 0 ? 1 : 0; },
      0, thrust::plus<int>());

  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept, ctx.queue);
  log_info("[AbsGS] Remove {} dead gaussians (kept {})", num_gaussians - nums_kept, nums_kept);
}

void AbsGSStrategy::set_params(const json& config) {
  StrategyBase::set_params(config);
  m_rng.seed(m_params.seed);
  if (config.contains("absgrad_threshold")) {
    m_absgrad_threshold = config["absgrad_threshold"].get<float>();
  }
  if (config.contains("percent_dense")) {
    m_percent_dense = config["percent_dense"].get<float>();
  }
}

json AbsGSStrategy::get_params() const {
  json params = StrategyBase::get_params();
  params["type"] = "absgs";
  params["absgrad_threshold"] = m_absgrad_threshold;
  params["percent_dense"] = m_percent_dense;
  return params;
}

}  // namespace tinygs
