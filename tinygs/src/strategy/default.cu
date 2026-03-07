#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/platform/buffer_utils.hpp"
#include "random/device.cuh"
#include "tinygs/strategy/default.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

DefaultStrategy::DefaultStrategy(
    std::shared_ptr<BackendRuntime> runtime,
    std::shared_ptr<GPUGaussian3d> gaussians,
    std::shared_ptr<GPUGaussian3d> gaussians_grad,
    std::shared_ptr<OptimizerBase> optimizer
) : StrategyBase(runtime, gaussians, gaussians_grad, optimizer) {
}

DefaultStrategy::~DefaultStrategy() = default;

void reset_opacity(const std::shared_ptr<GPUGaussian3d>& gaussians, float min_opacity_threshold, cudaStream_t stream) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(stream);

  thrust::for_each(
    exec,
    gaussians->opacities().begin(),
    gaussians->opacities().end(),
    [min_opacity_threshold] __device__ (float& opacity) {
      opacity = deactivate_opacity(fminf(activate_opacity(opacity), min_opacity_threshold));
    }
  );
}

void DefaultStrategy::step_impl(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue))); // make sure the operations on training stream are done.
  if (!ctx.densification_info) {
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(*ctx.runtime, num_gaussians);
    fill_buffer_zero_async(*ctx.runtime, *ctx.queue, ctx.densification_info);
  }

  auto step = this_step();
  if (step % m_params.refine_every == 0 &&
      step >= m_params.start_refine &&
      step <= m_params.end_refine) {
    if (m_gaussians->size() < m_params.max_num_gaussians) {
      duplicate(ctx);
    }
    // res contains marks the duplication gaussians, disable the pruning for them.
    prune(ctx);
    // after pruning, we need to reset the densification info since the indices have changed.
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(*ctx.runtime, num_gaussians);
    fill_buffer_zero_async(*ctx.runtime, *ctx.queue, ctx.densification_info);
  }

  if (m_params.reset_every > 0 && step % m_params.reset_every == 0 &&
      step >= m_params.start_refine && step < m_params.end_refine) {
    reset_opacity(m_gaussians, 2 * m_params.pruning_opacity_threshold, to_cuda_stream(ctx.queue));
    on_reset_opacity(ctx.queue);
  }
}

void DefaultStrategy::reset() {
  // Nothing to do here.
}

void DefaultStrategy::duplicate(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
  auto num_gaussians = m_gaussians->size();
  thrust::device_vector<char> duplication_flags(num_gaussians, 0);
  auto *d_grow_flags = thrust::raw_pointer_cast(duplication_flags.data());
  if (! ctx.densification_info || buffer_count<DensificationInfo>(ctx.densification_info) != num_gaussians) {
    log_warning("Densification info is not provided or has wrong size, skip duplication.");
    return; // no duplication happened
  }

  auto *d_densification_info = buffer_data<DensificationInfo>(ctx.densification_info);
  auto *d_scale = thrust::raw_pointer_cast(m_gaussians->scales().data());

  constexpr int kDuplicate = 1;
  constexpr int kSplit = 2;

#ifndef NDEBUG
  // Compute gradient statistics
  thrust::device_vector<float> gradient_values(num_gaussians);
  auto *d_gradient_values = thrust::raw_pointer_cast(gradient_values.data());
  
  // First pass: compute and store all gradient values
  thrust::for_each(                                       //
      exec,                                               //
      thrust::make_counting_iterator<int>(0),             //
      thrust::make_counting_iterator<int>(num_gaussians), //
      [d_densification_info, d_gradient_values,
       use_absgrad = m_params.absgrad
      ] __device__(int i) {
        const float accum = use_absgrad
                               ? d_densification_info[i].accum_absgrad_mean2d
                               : d_densification_info[i].accum_grad_mean2d;
        const float grad = accum / fmaxf(d_densification_info[i].accum_counter, 1.0f);
        d_gradient_values[i] = grad;
      } //
  );
  
  // Create device pointers for thrust algorithms
  thrust::device_ptr<float> grad_ptr = thrust::device_pointer_cast(d_gradient_values);
  thrust::device_ptr<float> grad_end = grad_ptr + num_gaussians;
  
  // Compute statistics using thrust algorithms
  auto minmax_result = thrust::minmax_element(exec, grad_ptr, grad_end);
  float min_val = *minmax_result.first;
  float max_val = *minmax_result.second;
  
  float mean = thrust::reduce(exec, grad_ptr, grad_end, 0.0f) / num_gaussians;
  
  // Compute standard deviation
  float variance = thrust::transform_reduce(
      exec,
      grad_ptr,
      grad_end,
      [mean] __host__ __device__(float x) { float diff = x - mean; return diff * diff; },
      0.0f,
      thrust::plus<float>()
  ) / num_gaussians;
  float std = sqrtf(variance);

  float nonzeros = thrust::transform_reduce(
      exec,
      grad_ptr,
      grad_end,
      [] __host__ __device__(float x) { return x > 0 ? 1 : 0; },
      0,
      thrust::plus<float>()
  );
  
  // Print gradient statistics
  printf("Gradient stats - Min: %.4g, Max: %.4g, Mean: %.4g, Std: %.4g, "
         "Nonzeros: %.4g, threshold: %.4g\n",
         min_val, max_val, mean, std, nonzeros, m_params.duplicate_grad_threshold * ctx.grad_scaler);
#endif

  thrust::for_each(                                       //
      exec,                                               //
      thrust::make_counting_iterator<int>(0),             //
      thrust::make_counting_iterator<int>(num_gaussians), //
      [d_densification_info, d_scale, d_grow_flags,
       grow_scale = m_params.duplicate_scale_threshold * m_gaussians->scene_scale(),
       //? The computed gradient is scaled by the scaler, we compensate this.
       grow_grad = m_params.duplicate_grad_threshold * ctx.grad_scaler,
       use_absgrad = m_params.absgrad
      ] __device__(int i) {
        const float accum = use_absgrad
                               ? d_densification_info[i].accum_absgrad_mean2d
                               : d_densification_info[i].accum_grad_mean2d;
        const float grad = accum / fmaxf(d_densification_info[i].accum_counter, 1.0f);
        if (grad > grow_grad && d_densification_info[i].accum_counter > 0) {
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
    exec,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] != 0 ? 1 : 0; },
    0,
    thrust::plus<int>()
  );
  const int num_dups = thrust::transform_reduce(
    exec,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] == kDuplicate ? 1 : 0; },
    0,
    thrust::plus<int>()
  );

  const int num_split = num_grows - num_dups;
  log_info("Add {} gaussians ({} total, {} split, {} duplicate)", num_grows, num_grows + num_gaussians,
      num_split, num_dups);

  if (num_grows == 0) return;

  thrust::device_vector<int> grow_indices_src(num_grows);
  auto *d_grow_indices_src = thrust::raw_pointer_cast(grow_indices_src.data());
  {
    auto * out = thrust::copy_if(                            //
        exec,                                                //
        thrust::make_counting_iterator<int>(0),              //
        thrust::make_counting_iterator<int>(num_gaussians),  //
        d_grow_flags, d_grow_indices_src, []__device__(char f) { return f != 0; });
    if (out - d_grow_indices_src != num_grows) {
      log_error("Grow source indices copy failed, expected {} but got {}",
                num_grows, out - d_grow_indices_src);
    }
  }

  thrust::device_vector<int> grow_indices_target(num_grows, 0);
  auto *d_grow_indices_target = thrust::raw_pointer_cast(grow_indices_target.data());
  auto* out = thrust::copy(
      exec,
      thrust::make_counting_iterator<int>(num_gaussians),
      thrust::make_counting_iterator<int>(num_gaussians + num_grows),
      d_grow_indices_target);

  StrategyBase::on_duplicate(d_grow_indices_src, d_grow_indices_target, num_grows, ctx.queue);
  // Now gaussians should have (num_gaussians + nums_duplicated) gaussians
  if (num_gaussians + num_grows != m_gaussians->size()) {
    log_error("Grow gaussians failed, expected {} gaussians, but got {}",
              num_gaussians + num_grows, m_gaussians->size());
  }


  thrust::device_vector<float> device_scales(num_grows * 6);
  generate_random_logistic(m_rng, num_grows * 6,
                           thrust::raw_pointer_cast(device_scales.data()),
                           (float)0.0, (float)1.0);

  // Do the duplicate and split.
  // After on_duplicate, SH buffers are in SoA layout with stride N_new = num_gaussians + num_grows.
  const int n_new = static_cast<int>(m_gaussians->size());
  thrust::for_each(exec,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_grows),
    [d_grow_indices_src, d_grow_indices_target, num_grows, d_grow_flags,
     means3d = thrust::raw_pointer_cast(m_gaussians->means().data()),
     scales3d = thrust::raw_pointer_cast(m_gaussians->scales().data()),
     opacities = thrust::raw_pointer_cast(m_gaussians->opacities().data()),
     rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
     sh0_data = thrust::raw_pointer_cast(m_gaussians->sh0().data()),
     sh1_data = thrust::raw_pointer_cast(m_gaussians->sh1().data()),
     sh2_data = thrust::raw_pointer_cast(m_gaussians->sh2().data()),
     sh3_data = thrust::raw_pointer_cast(m_gaussians->sh3().data()),
     n_new,
     device_scales = thrust::raw_pointer_cast(device_scales.data())
     ] __device__(int i) {
      int src_idx = d_grow_indices_src[i];
      int target_idx = d_grow_indices_target[i];
      if (d_grow_flags[src_idx] == 0) return;

      rotations[target_idx] = rotations[src_idx];
      // Copy SH data per-degree in SoA layout: for each channel, copy src→target at stride n_new
      for (int ch = 0; ch < 1 * 3; ch++) {  // sh0: 1 coeff * 3 channels
        sh0_data[ch * n_new + target_idx] = sh0_data[ch * n_new + src_idx];
      }
      for (int ch = 0; ch < 3 * 3; ch++) {  // sh1: 3 coeffs * 3 channels
        sh1_data[ch * n_new + target_idx] = sh1_data[ch * n_new + src_idx];
      }
      for (int ch = 0; ch < 5 * 3; ch++) {  // sh2: 5 coeffs * 3 channels
        sh2_data[ch * n_new + target_idx] = sh2_data[ch * n_new + src_idx];
      }
      for (int ch = 0; ch < 7 * 3; ch++) {  // sh3: 7 coeffs * 3 channels
        sh3_data[ch * n_new + target_idx] = sh3_data[ch * n_new + src_idx];
      }
      if (d_grow_flags[src_idx] == kDuplicate) {
        // keep everything same as src gs
        means3d[target_idx] = means3d[src_idx];
        scales3d[target_idx] = scales3d[src_idx];
        opacities[target_idx] = opacities[src_idx];
      } else {
        // NOTE: It is a little bit confusing, but (w, x, y, z) is a standard representation in
        // 3dgs, while (x, y, z, w) is the way to access it.
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
          2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
        );
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

void DefaultStrategy::prune(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));

  auto abs_ss_threshold = max(ctx.fwd_input.width, ctx.fwd_input.height) * m_params.max_screen_size;

  // Remove dead gaussians
  const auto num_gaussians = m_gaussians->size();
  const int original_num_gasussians = buffer_count<DensificationInfo>(ctx.densification_info);
  thrust::device_vector<char> is_alive(num_gaussians);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  thrust::for_each(exec,                                                     //
      thrust::make_counting_iterator<int>(0),                                //
      thrust::make_counting_iterator<int>(num_gaussians),                    //
      [d_is_alive = is_alive.data(), d_opacity,                              //
       scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),       //
       scene_scale = m_gaussians->scene_scale(),                             //
       rotation = thrust::raw_pointer_cast(m_gaussians->rotations().data()), //
       pruning_scale_threshold = m_params.pruning_scale_threshold,           //
       prune_large = this_step() > m_params.reset_every,                     //
       max_radii_screen_threshold = abs_ss_threshold,                        //
       original_num_gasussians,
       deninfo = buffer_data<DensificationInfo>(ctx.densification_info),                 //
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) { //
        bool not_large_ws = max(activate_scale(scale[i])) < pruning_scale_threshold * scene_scale;
        bool not_large_ss = i >= original_num_gasussians || deninfo[i].max_radii_screen < max_radii_screen_threshold;
        bool not_transparent = activate_opacity(d_opacity[i]) > min_opacity;
        bool not_degenerate = sum(abs(rotation[i])) > FLT_EPSILON;

        if (not_transparent && ((not_large_ws && not_large_ss) || !prune_large) && not_degenerate) {
          d_is_alive[i] = 1;
        } else {
          d_is_alive[i] = 0;
        }
      });

  int nums_kept = thrust::transform_reduce(
    exec,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [ia = is_alive.data()] __device__(int i) -> int { return ia[i] != 0 ? 1 : 0; },
    0,
    thrust::plus<int>()
  );

  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept, ctx.queue);
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
