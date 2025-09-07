#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/strategy/default.hpp"
#include "tinygs/utils/scope_timer.hpp"

namespace tinygs {

DefaultStrategy::~DefaultStrategy() = default;

void DefaultStrategy::step_impl(const RasterizeContext& ctx) {
  TINYGS_TIMER("DefaultStrategy::step");

  auto step = this_step();
  if (step % m_params.refine_every == 0 &&
      step >= m_params.start_refine &&
      step <= m_params.end_refine) {
    const auto dup_flag = duplicate(ctx);
    // res contains marks the duplication gaussians, disable the pruning for them.
    prune(ctx, dup_flag);
    // after pruning, we need to reset the densification info since the indices have changed.
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = std::make_shared<GPUBuffer<float>>(num_gaussians * 2);
    ctx.densification_info->memset(0);
    ctx.radii.resize(m_gaussians->size());
    thrust::fill(ctx.radii.begin(), ctx.radii.end(), 0);
  }
}

void DefaultStrategy::reset() {
  // TODO: implement reset
}

thrust::device_vector<bool> DefaultStrategy::duplicate(const RasterizeContext& ctx) {
  // TODO: implement the split and duplicate (grow_gs)

  auto num_gaussians = m_gaussians->size();
  // GPUBuffer<char> duplication_flags(ctx.stream, num_gaussians);
  thrust::device_vector<char> duplication_flags(num_gaussians);
  auto *d_grow_flags = thrust::raw_pointer_cast(duplication_flags.data());
  if (! ctx.densification_info || ctx.densification_info->size() != num_gaussians * 2) {
    log_warning("Densification info is not provided or has wrong size, skip duplication.");
    return thrust::device_vector<bool>(num_gaussians, false); // no duplication happened
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
          const float max_scale = expf(max(d_scale[i]));
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


  // TODO: replace the seed with global defined.
  thrust::default_random_engine rng(42);
  thrust::normal_distribution<float> dist(0.f, 1.f);
  thrust::host_vector<float> host_scales(num_grows * 6);
  thrust::generate(host_scales.begin(), host_scales.end(), [&] { return dist(rng); });
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
      assert(0 <= src_idx && src_idx < num_gaussians);
      assert(num_gaussians <= target_idx && target_idx < num_gaussians + num_grows);
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
        const vec3 actual_scale = exp(scales3d[src_idx]);
        const float new_opacity = 1.0f - sqrtf(1.0f - logistic(opacities[src_idx]));
        const vec3 rand1 = vec3(device_scales[i * 6 + 0], device_scales[i * 6 + 1], device_scales[i * 6 + 2]);
        const vec3 rand2 = vec3(device_scales[i * 6 + 3], device_scales[i * 6 + 4], device_scales[i * 6 + 5]);
        const vec3 off1 = rot * (actual_scale * rand1);
        const vec3 off2 = rot * (actual_scale * rand2);

        /// 1. target gs
        means3d[target_idx] = means3d[src_idx] + off1;
        scales3d[target_idx] = log(actual_scale / 1.6f);
        opacities[target_idx] = logit(new_opacity);

        /// 2. src gs
        means3d[src_idx] = means3d[src_idx] + off2;
        scales3d[src_idx] = log(actual_scale / 1.6f);
        opacities[src_idx] = logit(new_opacity);
      }
    }
  );

  thrust::device_vector<bool> last_duplications(m_gaussians->size());
  thrust::transform(
    thrust::device,
    duplication_flags.begin(),
    duplication_flags.end(),
    last_duplications.begin(),
    [] __device__ (char f) {return f != 0; }
  );

  thrust::copy(
    thrust::make_counting_iterator<int>(num_gaussians),
    thrust::make_counting_iterator<int>(num_grows + num_gaussians),
    last_duplications.begin() + num_gaussians);

  return last_duplications;
}

void DefaultStrategy::prune(const RasterizeContext& ctx, const thrust::device_vector<bool> & disable_prune) {
  // Remove dead gaussians
  const auto num_gaussians = m_gaussians->size();
  thrust::device_vector<char> is_alive(num_gaussians);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  thrust::for_each(                                                          //
      thrust::make_counting_iterator<int>(0),                                //
      thrust::make_counting_iterator<int>(num_gaussians),                    //
      [d_is_alive = is_alive.data(), d_opacity,                              //
       d_disable = disable_prune.data(),                                     //
       scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),       //
       scene_scale = m_gaussians->scene_scale(),                             //
       pruning_scale_threshold = m_params.pruning_scale_threshold,           //
       prune_large = this_step() > m_params.reset_every,                     //
       radii = thrust::raw_pointer_cast(ctx.radii.data()),                   //
       max_radii = ctx.radii.size(),                                         //
       max_radii_threshold = m_params.max_screen_size,                       //
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) { //
        bool not_large_ws = max(exp(scale[i])) < pruning_scale_threshold * scene_scale;
        bool not_large_vs = i < max_radii || radii[i] < max_radii_threshold;
        bool not_transparent = logistic(d_opacity[i]) > min_opacity;

        if (not_transparent && (not_large_ws && not_large_vs || !prune_large)) {
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

DefaultStrategy::DefaultStrategy(std::shared_ptr<GPUGaussian3d> gaussians)
    : StrategyBase(gaussians) {}

} // namespace tinygs