#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

#include "rasterizer/fastgs_ours/utils.h"
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/strategy/default.hpp"
#include "utils/scope_timer.hpp"
namespace tinygs {

DefaultStrategy::~DefaultStrategy() = default;

void DefaultStrategy::step(const RasterizeContext& ctx) {
  // TODO: implement step
  TINYGS_TIMER("DefaultStrategy::step");

  // prune(ctx);
  duplicate(ctx);
  CUDA_CHECK_THROW(cudaDeviceSynchronize());
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
    return;
  }

  auto *d_densification_info = ctx.densification_info->data();
  auto *d_scale = thrust::raw_pointer_cast(m_gaussians->scales().data());

  constexpr int kDuplicate = 1;
  constexpr int kSplit = 2;

  thrust::for_each(                                       //
      thrust::make_counting_iterator<int>(0),             //
      thrust::make_counting_iterator<int>(num_gaussians), //
      [d_densification_info, d_scale, num_gaussians, d_grow_flags,
       grow_scale = m_params.duplicate_scale_threshold,
       grow_grad = m_params.duplicate_grad_threshold] __device__(int i) {
        const float grad = d_densification_info[i + num_gaussians] /
                           fmaxf(d_densification_info[i], 1.0f);
        if (grad > grow_grad) {
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

  int num_grows = thrust::transform_reduce(
    thrust::device,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] != 0 ? 1 : 0; },
    0,
    thrust::plus<int>()
  );

  log_info("Duplicate {} gaussians ({} total)", num_grows, num_grows + num_gaussians);

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

  this->post_duplicate(d_grow_indices_src, d_grow_indices_target, num_grows);
  // Now gaussians should have (num_gaussians + nums_duplicated) gaussians
  if (num_gaussians + num_grows != m_gaussians->size()) {
    log_error("Grow gaussians failed, expected {} gaussians, but got {}",
              num_gaussians + num_grows, m_gaussians->size());
  }


  // TODO: replace the seed with global defined.
  thrust::default_random_engine rng(1337);
  thrust::normal_distribution<float> dist;
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
      assert(0 <= target_idx && target_idx < num_gaussians + num_grows);
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
        const mat3x3 rot = to_mat3(normalize(quat{              //
          rotations[src_idx].x, rotations[src_idx].y, //
          rotations[src_idx].z, rotations[src_idx].w  //
        }));
        const vec3 actual_scale = exp(scales3d[src_idx]);
        const float new_opacity = 1.0f - sqrtf(1 - logistic(opacities[src_idx]));
        const vec3 randn = vec3(device_scales[i * 3 + 0], device_scales[i * 3 + 1], device_scales[i * 3 + 2]);

        /// 1. target gs
        means3d[target_idx] = means3d[src_idx] + (rot * randn) * actual_scale;
        scales3d[target_idx] = log(actual_scale / 1.6f);
        opacities[target_idx] = logit(new_opacity);

        /// 2. src gs
        means3d[src_idx] = means3d[src_idx] - (rot * randn) * actual_scale;
        scales3d[src_idx] = log(actual_scale / 1.6f);
        opacities[src_idx] = logit(new_opacity);
      }
    }
  );
}

void DefaultStrategy::prune(const RasterizeContext& /* ctx */) {
  // Remove dead gaussians
  const auto num_gaussians = m_gaussians->size();
  // GPUBuffer<char> is_alive(num_gaussians);
  thrust::device_vector<char> is_alive(num_gaussians);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  const auto* d_rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data());

  thrust::for_each(                                                          //
      thrust::make_counting_iterator<int>(0),                                //
      thrust::make_counting_iterator<int>(num_gaussians),                    //
      [d_is_alive = is_alive.data(), d_opacity, d_rotations,                 //
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) { //
        if (logistic(d_opacity[i]) > min_opacity && length2(d_rotations[i]) > 1.0e-8f) {
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

  this->remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept);
  log_info("Remove {} dead gaussians (kept {})", num_gaussians - nums_kept, nums_kept);
}

DefaultStrategy::DefaultStrategy(std::shared_ptr<GPUGaussian3d> gaussians)
    : StrategyBase(gaussians) {}

} // namespace tinygs