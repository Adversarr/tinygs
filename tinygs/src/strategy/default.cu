#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/strategy/default.hpp"
#include "tinygs/utils/scope_timer.hpp"

namespace tinygs {

DefaultStrategy::DefaultStrategy(
    std::shared_ptr<GPUGaussian3d> gaussians,
    std::shared_ptr<GPUGaussian3d> gaussians_grad,
    std::shared_ptr<OptimizerBase> optimizer
) : StrategyBase(gaussians, gaussians_grad, optimizer) {
}

DefaultStrategy::~DefaultStrategy() = default;

void reset_opacity(const std::shared_ptr<GPUGaussian3d>& gaussians, float min_opacity_threshold) {
  TINYGS_TIMER("DefaultStrategy::reset_opacity");

  thrust::for_each(
    thrust::device,
    gaussians->opacities().begin(),
    gaussians->opacities().end(),
    [min_opacity_threshold] __device__ (float& opacity) {
      opacity = deactivate_opacity(fminf(activate_opacity(opacity), min_opacity_threshold));
    }
  );
}

void DefaultStrategy::step_impl(const RasterizeContext& ctx) {
  TINYGS_TIMER("DefaultStrategy::step");
  if (!ctx.densification_info) {
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = std::make_shared<GPUBuffer<float>>(num_gaussians * 2);
    ctx.densification_info->memset(0);
  }

  auto step = this_step();
  if (step % m_params.refine_every == 0 &&
      step >= m_params.start_refine &&
      step <= m_params.end_refine) {
    duplicate(ctx);
    // res contains marks the duplication gaussians, disable the pruning for them.
    prune(ctx);
    // after pruning, we need to reset the densification info since the indices have changed.
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = std::make_shared<GPUBuffer<float>>(num_gaussians * 2);
    ctx.densification_info->memset(0);
  }

  if (step % m_params.reset_every == 0 && step >= m_params.start_refine && step <= m_params.end_refine) {
    reset_opacity(m_gaussians, 2 * m_params.pruning_opacity_threshold);
    on_reset_opacity();
  }
}

void DefaultStrategy::reset() {
  // TODO: implement reset
}

void DefaultStrategy::duplicate(const RasterizeContext& ctx) {
  // TODO: implement the split and duplicate (grow_gs)

  auto num_gaussians = m_gaussians->size();
  // GPUBuffer<char> duplication_flags(ctx.stream, num_gaussians);
  thrust::device_vector<char> duplication_flags(num_gaussians);
  auto *d_grow_flags = thrust::raw_pointer_cast(duplication_flags.data());
  if (! ctx.densification_info || ctx.densification_info->size() != num_gaussians * 2) {
    log_warning("Densification info is not provided or has wrong size, skip duplication.");
    return; // no duplication happened
  }

  auto *d_densification_info = ctx.densification_info->data();
  auto *d_scale = thrust::raw_pointer_cast(m_gaussians->scales().data());

  constexpr int kDuplicate = 1;
  constexpr int kSplit = 2;

  thrust::for_each(                                       //
      thrust::make_counting_iterator<int>(0),             //
      thrust::make_counting_iterator<int>(num_gaussians), //
      [d_densification_info, d_scale, num_gaussians, d_grow_flags,
       grow_scale = m_params.duplicate_scale_threshold * m_gaussians->scene_scale(),
       grow_grad = m_params.duplicate_grad_threshold] __device__(int i) {
        const float grad = d_densification_info[i + num_gaussians] /
                           fmaxf(d_densification_info[i], 1.0f);
        if (grad > grow_grad && d_densification_info[i] > 0) {
          const float max_scale = max(activate_scale(d_scale[i]));
          if (max_scale > grow_scale) { // is_large => split
            d_grow_flags[i] = kSplit;
          } else {
            d_grow_flags[i] = kDuplicate;
          }
        } else {
          d_grow_flags[i] = 0;
        }
      } //
  );

  const int num_grows = thrust::transform_reduce(
    thrust::device,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] != 0 ? 1 : 0; },
    0,
    thrust::plus<int>()
  );
  const int num_dups = thrust::transform_reduce(
    thrust::device,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] == kDuplicate ? 1 : 0; },
    0,
    thrust::plus<int>()
  );

  const int num_split = num_grows - num_dups;
  log_info("Add {} gaussians ({} total, {} split, {} duplicate)", num_grows, num_grows + num_gaussians,
      num_split, num_dups);

  thrust::device_vector<int> grow_indices_src(num_grows);
  auto *d_grow_indices_src = thrust::raw_pointer_cast(grow_indices_src.data());
  {
    auto * out = thrust::copy_if(                            //
        thrust::device,                                      //
        thrust::make_counting_iterator<int>(0),              //
        thrust::make_counting_iterator<int>(num_gaussians),  //
        d_grow_flags, d_grow_indices_src, []__device__(char f) { return f != 0; });
    if (out - d_grow_indices_src != num_grows) {
      log_error("Grow source indices copy failed, expected {} but got {}",
                num_grows, out - d_grow_indices_src);
    }
  }

  thrust::device_vector<int> grow_indices_target(num_grows);
  auto *d_grow_indices_target = thrust::raw_pointer_cast(grow_indices_target.data());
  {
    auto* out = thrust::copy(
       thrust::device,
       thrust::make_counting_iterator<int>(num_gaussians),
       thrust::make_counting_iterator<int>(num_gaussians + num_grows),
       d_grow_indices_target);
    if (out - d_grow_indices_target != num_grows) {
      log_error("Grow target indices copy failed, expected {} but got {}",
                num_grows, out - d_grow_indices_target);
    }
  }

  StrategyBase::on_duplicate(d_grow_indices_src, d_grow_indices_target, num_grows);
  // Now gaussians should have (num_gaussians + nums_duplicated) gaussians
  if (num_gaussians + num_grows != m_gaussians->size()) {
    log_error("Grow gaussians failed, expected {} gaussians, but got {}",
              num_gaussians + num_grows, m_gaussians->size());
  }


  thrust::normal_distribution<float> dist(0.f, 1.f);
  thrust::host_vector<float> host_scales(num_grows * 6);
  thrust::generate(host_scales.begin(), host_scales.end(), [&] {
    float u1 = 1 - m_rng.next_float();
    float u2 = m_rng.next_float();
    // Box-Muller transform with safety checks
    const float epsilon = 1e-7f;
    u1 = std::max(epsilon, std::min(1.0f - epsilon, u1)); // Ensure u1 is in (0,1)
    const float noise = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
    return std::isfinite(noise) ? noise : 0.0f; // Return 0 if result is invalid
  });

  thrust::device_vector<float> device_scales = host_scales;

  // Do the duplicate and split.
  thrust::for_each(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_grows),
    [d_grow_indices_src, d_grow_indices_target, num_grows, d_grow_flags, num_gaussians,
     means3d = thrust::raw_pointer_cast(m_gaussians->means().data()),
     scales3d = thrust::raw_pointer_cast(m_gaussians->scales().data()),
     opacities = thrust::raw_pointer_cast(m_gaussians->opacities().data()),
     rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
     sh0 = thrust::raw_pointer_cast(m_gaussians->sh_coefficient_0().data()),
     sh_rest = thrust::raw_pointer_cast(m_gaussians->sh_coefficients_rest().data()),
     device_scales = thrust::raw_pointer_cast(device_scales.data())
     ] __device__(int i) {
      int src_idx = d_grow_indices_src[i];
      int target_idx = d_grow_indices_target[i];
      if (d_grow_flags[src_idx] == 0) return;

      rotations[target_idx] = rotations[src_idx];
      sh0[target_idx] = sh0[src_idx];
      for (int i = 0; i <= 15; i++) {
        sh_rest[target_idx * 15 + i] = sh_rest[src_idx * 15 + i];
      }
      if (d_grow_flags[src_idx] == kDuplicate) {
        // keep everything same as src gs
        means3d[target_idx] = means3d[src_idx];
        scales3d[target_idx] = scales3d[src_idx];
        opacities[target_idx] = opacities[src_idx];
      } else {
        const mat3x3 rot = quat_to_mat3(normalize(quat{    //
          rotations[src_idx].x, rotations[src_idx].y, //
          rotations[src_idx].z, rotations[src_idx].w  //
        }));
        const vec3 actual_scale = activate_scale(scales3d[src_idx]);
        const float new_opacity = 1.0f - sqrtf(1.0f - activate_opacity(opacities[src_idx]));
        const vec3 rand1 = vec3(device_scales[i * 6 + 0], device_scales[i * 6 + 1], device_scales[i * 6 + 2]);
        const vec3 rand2 = vec3(device_scales[i * 6 + 3], device_scales[i * 6 + 4], device_scales[i * 6 + 5]);
        const vec3 off1 = rot * (rand1 * (actual_scale + 1e-5f));
        const vec3 off2 = rot * (rand2 * (actual_scale + 1e-5f));

        /// 1. target gs
        means3d[target_idx] = means3d[src_idx] + off1;
        scales3d[target_idx] = deactivate_scale(actual_scale / 1.6f);
        opacities[target_idx] = deactivate_opacity(new_opacity);

        /// 2. src gs
        means3d[src_idx] = means3d[src_idx] + off2;
        scales3d[src_idx] = deactivate_scale(actual_scale / 1.6f);
        opacities[src_idx] = deactivate_opacity(new_opacity);
      }
    }
  );
}

void DefaultStrategy::prune(const RasterizeContext& /* ctx */) {
  // Remove dead gaussians
  const auto num_gaussians = m_gaussians->size();
  thrust::device_vector<char> is_alive(num_gaussians);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  thrust::for_each(                                                          //
      thrust::make_counting_iterator<int>(0),                                //
      thrust::make_counting_iterator<int>(num_gaussians),                    //
      [d_is_alive = is_alive.data(), d_opacity,                              //
       scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),       //
       scene_scale = m_gaussians->scene_scale(),                             //
       pruning_scale_threshold = m_params.pruning_scale_threshold,           //
       prune_large = this_step() > m_params.reset_every,                     //
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) { //
        bool not_large_ws = max(activate_scale(scale[i])) < pruning_scale_threshold * scene_scale;
        bool not_transparent = activate_opacity(d_opacity[i]) > min_opacity;

        if (not_transparent && (not_large_ws || !prune_large)) {
          d_is_alive[i] = 1;
        } else {
          d_is_alive[i] = 0;
        }
      });

  int nums_kept = thrust::transform_reduce(
    thrust::device,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [ia = is_alive.data()] __device__(int i) -> int { return ia[i] != 0 ? 1 : 0; },
    0,
    thrust::plus<int>()
  );

  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept);
  log_info("Remove {} dead gaussians (kept {})", num_gaussians - nums_kept, nums_kept);
}



void DefaultStrategy::set_params(const json& config) {
  StrategyBase::set_params(config);
  m_rng.seed(m_params.seed);
}

json DefaultStrategy::get_params() const {
  json params = StrategyBase::get_params();
  params["type"] = "default";
  return params;
}

} // namespace tinygs