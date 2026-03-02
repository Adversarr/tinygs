// FastGS densification strategy.
//
// Multi-view metric scoring: renders M random cameras, computes per-pixel L1 loss vs GT,
// thresholds at `loss_thresh` to get a binary metric map, then counts how many flagged
// pixels each Gaussian contributes to.  These counts become `importance_score`.
//
// Clone: grad >= clone_thresh AND scale <= percent_dense * scene_scale AND importance > threshold
// Split: absgrad >= absgrad_thresh AND scale > percent_dense * scene_scale AND importance > threshold
// Prune: budget-based multinomial sampling with weight = 1 / (1e-6 + 1 - pruning_score)
// Final prune: opacity < 0.1 OR pruning_score > 0.9
//
// Reference: ref_impl/FastGS/scene/gaussian_model.py

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <nvtx3/nvtx3.hpp>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/random/device.cuh"
#include "tinygs/strategy/fastgs.hpp"

namespace tinygs {

FastGSStrategy::FastGSStrategy(
    std::shared_ptr<GPUGaussian3d> gaussians,
    std::shared_ptr<GPUGaussian3d> gaussians_grad,
    std::shared_ptr<OptimizerBase> optimizer)
    : StrategyBase(gaussians, gaussians_grad, optimizer) {}

FastGSStrategy::~FastGSStrategy() = default;

void FastGSStrategy::set_rasterizer(std::shared_ptr<RasterizerBase> rasterizer) {
  m_rasterizer = std::move(rasterizer);
}

void FastGSStrategy::set_dataloader(std::shared_ptr<DataLoaderBase> dataloader) {
  m_dataloader = std::move(dataloader);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

/// @brief Compute per-pixel mean L1 loss across 3 channels and produce a binary metric map.
///        metric_map[pixel] = 1 if mean_L1 > loss_thresh, 0 otherwise.
///        Supports Float32 (CHW padded) images.
__global__ void compute_metric_map_kernel(
    int n_pixels,
    const float* __restrict__ rendered,    // CHW padded image (rendered)
    const float* __restrict__ gt,          // CHW padded image (ground truth)
    int* __restrict__ metric_map,          // H*W output binary map
    int padded_w,                          // padded width (pixels per row per channel)
    int padded_h,                          // padded height
    int width,                             // actual width
    int height,                            // actual height
    float loss_thresh) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pixels) return;
  const int py = idx / width;
  const int px = idx % width;
  if (py >= height) return;

  // Accumulate L1 across 3 channels in CHW layout
  float l1_sum = 0.0f;
  for (int c = 0; c < 3; c++) {
    const int offset = c * padded_w * padded_h + py * padded_w + px;
    l1_sum += fabsf(fminf(fmaxf(rendered[offset], 0.0f), 1.0f) - gt[offset]);
  }
  metric_map[idx] = (l1_sum / 3.0f > loss_thresh) ? 1 : 0;
}

/// @brief Accumulate importance and pruning scores from densification info.
///        importance_score += metric_importance_score / num_cameras  (floored)
///        pruning_score = metric_pruning_score  (overwritten each time)
__global__ void accumulate_scores_kernel(
    int n,
    const DensificationInfo* __restrict__ den_info,
    float* __restrict__ importance_score,
    float* __restrict__ pruning_score,
    int num_cameras) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  importance_score[idx] = floorf(den_info[idx].metric_importance_score / static_cast<float>(num_cameras));
  pruning_score[idx] = den_info[idx].metric_pruning_score;
}

/// @brief Clamp opacity to max_value after densification.
__global__ void clamp_opacity_kernel(int n, float* __restrict__ opacities, float max_value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  opacities[idx] = deactivate_opacity(fminf(activate_opacity(opacities[idx]), max_value));
}

// ---------------------------------------------------------------------------
// compute_gaussian_score()
// ---------------------------------------------------------------------------

void FastGSStrategy::compute_gaussian_score(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();

  if (!m_rasterizer || !m_dataloader) {
    log_warning("[FastGS] Rasterizer or dataloader not set; skipping metric computation.");
    return;
  }

  const size_t num_gaussians = m_gaussians->size();
  auto dataset = m_dataloader->get_dataset();
  const size_t dataset_size = dataset->size();
  if (dataset_size == 0) {
    log_warning("[FastGS] Dataset is empty; skipping metric computation.");
    return;
  }

  // Allocate per-Gaussian cumulative counts
  thrust::device_vector<int> total_counts(num_gaussians, 0);
  auto* d_total_counts = thrust::raw_pointer_cast(total_counts.data());

  // We need a temporary RasterizeContext for rendering
  RasterizeContext metric_ctx;
  metric_ctx.inference = true;  // Don't save intermediates for backward
  metric_ctx.stream = ctx.stream;
  metric_ctx.grad_scaler = ctx.grad_scaler;
  metric_ctx.metric_mode = true;

  // Number of cameras to render
  const int num_cameras = std::min(m_metric_num_cameras, static_cast<int>(dataset_size));

  for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
    // Pick a random camera
    const size_t idx = m_rng.next_uint(static_cast<uint32_t>(dataset_size));
    auto data = (*dataset)[idx];

    // Set up input
    metric_ctx.fwd_input.width = data.image.shape.width;
    metric_ctx.fwd_input.height = data.image.shape.height;
    metric_ctx.fwd_input.K = data.K;
    metric_ctx.fwd_input.w2c = data.w2c;
    metric_ctx.fwd_input.near = ctx.fwd_input.near;
    metric_ctx.fwd_input.far = ctx.fwd_input.far;
    metric_ctx.fwd_input.timestamp = data.timestamp;

    // Allocate metric map and counts for this camera
    const int width = data.image.shape.width;
    const int height = data.image.shape.height;
    metric_ctx.metric_map = std::make_shared<GPUBuffer<int>>(width * height);
    metric_ctx.metric_counts = std::make_shared<GPUBuffer<int>>(num_gaussians);
    metric_ctx.metric_counts->memset(0);

    // Forward render (metric mode)
    m_rasterizer->forward_metric(metric_ctx);
    CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));

    // Compute per-pixel L1 loss between rendered and ground truth
    // We need to transfer the GT image to GPU
    const auto& rendered_image = metric_ctx.fwd_output.image;
    const int padded_w = rendered_image.shape.padded_width();
    const int padded_h = rendered_image.shape.padded_height();

    // Transfer GT to GPU
    GPUMemory<float> gt_gpu(padded_w * padded_h * 3);
    Image gt_image;
    gt_image.shape = rendered_image.shape;
    gt_image.data_type = DataType::Float32;
    gt_image.data = gt_gpu.data();
    m_dataloader->transfer_gpu(ctx.stream, gt_image, data.image);
    CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));

    // Compute metric map (binary threshold on L1)
    const int n_pixels = width * height;
    linear_kernel(compute_metric_map_kernel, 0, ctx.stream, n_pixels,
        static_cast<const float*>(rendered_image.data),
        static_cast<const float*>(gt_image.data),
        metric_ctx.metric_map->data(),
        padded_w, padded_h, width, height, m_loss_thresh);

    // The forward_metric pass may have accumulated counts already if the rasterizer
    // supports native metric accumulation.  If not (default forward delegation),
    // we fall back to a simpler approach: just count flagged pixels per Gaussian
    // via the densification info's metric fields updated by the rasterizer.
    //
    // For now, we simply accumulate the metric_counts from the rasterizer output.
    // If the rasterizer doesn't populate metric_counts, this will be zeros, and
    // importance_score stays 0 — effectively disabling FastGS filtering (fallback
    // to gradient-only densification).

    // Accumulate total_counts += metric_counts for this camera
    auto exec = thrust::cuda::par.on(ctx.stream);
    thrust::transform(exec,
        total_counts.begin(), total_counts.end(),
        thrust::device_pointer_cast(metric_ctx.metric_counts->data()),
        total_counts.begin(),
        thrust::plus<int>());
  }

  // Convert total_counts to per-Gaussian importance and pruning scores
  // importance_score = floor(total_counts / num_cameras)
  m_importance_score.resize(num_gaussians);
  m_pruning_score.resize(num_gaussians);

  auto exec = thrust::cuda::par.on(ctx.stream);
  const float inv_cams = 1.0f / static_cast<float>(num_cameras);
  thrust::transform(exec,
      total_counts.begin(), total_counts.end(),
      m_importance_score.begin(),
      [inv_cams] __device__(int count) -> float { return floorf(count * inv_cams); });

  // Pruning score: use metric_pruning_score from densification info if available,
  // otherwise derive from total_counts (higher counts → more important → lower prune score)
  if (ctx.densification_info) {
    const auto* den = ctx.densification_info->data();
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        [den, ps = thrust::raw_pointer_cast(m_pruning_score.data())] __device__(int i) {
          ps[i] = den[i].metric_pruning_score;
        });
  } else {
    thrust::fill(exec, m_pruning_score.begin(), m_pruning_score.end(), 0.0f);
  }

  CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));
}

// ---------------------------------------------------------------------------
// step_impl()
// ---------------------------------------------------------------------------

void FastGSStrategy::step_impl(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  CUDA_CHECK_THROW(cudaStreamSynchronize(ctx.stream));

  if (!ctx.densification_info) {
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = std::make_shared<GPUBuffer<DensificationInfo>>(num_gaussians);
    ctx.densification_info->memset(0);
  }

  const int step = this_step();
  if (step % m_params.refine_every == 0 &&
      step >= m_params.start_refine &&
      step <= m_params.end_refine) {

    // Compute multi-view importance scores before densification
    compute_gaussian_score(ctx);

    if (m_gaussians->size() < m_params.max_num_gaussians) {
      duplicate(ctx);
    }
    prune(ctx);

    // Clamp opacity after densification (FastGS specific)
    linear_kernel(clamp_opacity_kernel, 0, ctx.stream,
        static_cast<int>(m_gaussians->size()),
        thrust::raw_pointer_cast(m_gaussians->opacities().data()),
        m_opacity_reset_value);

    // Reset densification info since indices have changed
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = std::make_shared<GPUBuffer<DensificationInfo>>(num_gaussians);
    ctx.densification_info->memset(0);
    m_importance_score.clear();
    m_pruning_score.clear();
  }

  // Final prune: remove Gaussians with high pruning_score or low opacity
  if (step >= m_final_prune_start && step <= m_final_prune_end &&
      step % m_final_prune_every == 0) {
    // Need to recompute scores for final prune
    if (m_importance_score.empty()) {
      compute_gaussian_score(ctx);
    }
    final_prune(ctx);
  }

  if (m_params.reset_every > 0 && step % m_params.reset_every == 0 &&
      step >= m_params.start_refine && step < m_params.end_refine) {
    on_reset_opacity();
  }
}

void FastGSStrategy::reset() {
  m_importance_score.clear();
  m_pruning_score.clear();
}

// ---------------------------------------------------------------------------
// duplicate()
// ---------------------------------------------------------------------------

void FastGSStrategy::duplicate(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(ctx.stream);
  const auto num_gaussians = m_gaussians->size();

  if (!ctx.densification_info || ctx.densification_info->size() != num_gaussians) {
    log_warning("[FastGS] Densification info is not provided or has wrong size, skip duplication.");
    return;
  }

  const bool has_importance = (m_importance_score.size() == num_gaussians);

  GPUBuffer<char> grow_flags(ctx.stream, num_gaussians);
  grow_flags.memset_async(ctx.stream, 0);
  auto* d_grow_flags = thrust::raw_pointer_cast(grow_flags.data());
  auto* d_densification_info = ctx.densification_info->data();
  auto* d_scale = thrust::raw_pointer_cast(m_gaussians->scales().data());

  constexpr int kClone = 1;
  constexpr int kSplit = 2;

  const float scene_scale = m_gaussians->scene_scale();
  const float clone_thresh = m_params.duplicate_grad_threshold * ctx.grad_scaler;
  const float split_thresh = m_absgrad_threshold * ctx.grad_scaler;
  const float scale_boundary = m_percent_dense * scene_scale;
  const float importance_thresh = m_importance_threshold;
  const float* d_importance = has_importance ?
      thrust::raw_pointer_cast(m_importance_score.data()) : nullptr;

  thrust::for_each(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_densification_info, d_scale, d_grow_flags, d_importance,
       clone_thresh, split_thresh, scale_boundary, importance_thresh] __device__(int i) {
        const float counter = fmaxf(d_densification_info[i].accum_counter, 1.0f);
        const float grad = d_densification_info[i].accum_grad_mean2d / counter;
        const float absgrad = d_densification_info[i].accum_absgrad_mean2d / counter;
        const float max_scale = max(activate_scale(d_scale[i]));

        // FastGS: importance score filtering
        bool importance_ok = (d_importance == nullptr) ||
                              (d_importance[i] > importance_thresh);

        if (counter > 0 && importance_ok) {
          if (grad >= clone_thresh && max_scale <= scale_boundary) {
            d_grow_flags[i] = kClone;
          } else if (absgrad >= split_thresh && max_scale > scale_boundary) {
            d_grow_flags[i] = kSplit;
          }
        }
      });

  const int num_grows = thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] != 0 ? 1 : 0; },
      0, thrust::plus<int>());

  if (num_grows == 0) return;

  const int num_clones = thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] == kClone ? 1 : 0; },
      0, thrust::plus<int>());

  log_info("[FastGS] Add {} gaussians ({} clone, {} split, {} total)",
           num_grows, num_clones, num_grows - num_clones, num_grows + num_gaussians);

  // Collect source indices
  thrust::device_vector<int> grow_indices_src(num_grows);
  auto* d_grow_indices_src = thrust::raw_pointer_cast(grow_indices_src.data());
  thrust::copy_if(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      d_grow_flags, d_grow_indices_src,
      [] __device__(char f) { return f != 0; });

  // Target indices
  GPUBuffer<int> grow_indices_target(ctx.stream, num_grows);
  auto* d_grow_indices_target = thrust::raw_pointer_cast(grow_indices_target.data());
  thrust::copy(exec,
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians) + num_grows),
      d_grow_indices_target);

  StrategyBase::on_duplicate(d_grow_indices_src, d_grow_indices_target, num_grows);

  // Generate random samples for split offsets
  GPUBuffer<float> device_rng(ctx.stream, num_grows * 6);
  generate_random_logistic(m_rng, num_grows * 6,
      thrust::raw_pointer_cast(device_rng.data()), 0.0f, 1.0f);

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
          means3d[target_idx] = means3d[src_idx];
          scales3d[target_idx] = scales3d[src_idx];
          opacities[target_idx] = opacities[src_idx];
        } else {
          // Split along covariance
          const float r = rotations[src_idx].x;
          const float x = rotations[src_idx].y;
          const float y = rotations[src_idx].z;
          const float z = rotations[src_idx].w;
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

          means3d[target_idx] = means3d[src_idx] + off1;
          scales3d[target_idx] = deactivate_scale(actual_scale / 1.6f);
          opacities[target_idx] = deactivate_opacity(new_opacity);

          means3d[src_idx] = means3d[src_idx] + off2;
          scales3d[src_idx] = deactivate_scale(actual_scale / 1.6f);
          opacities[src_idx] = deactivate_opacity(new_opacity);
        }
      });
}

// ---------------------------------------------------------------------------
// prune()  — budget-based pruning
// ---------------------------------------------------------------------------

void FastGSStrategy::prune(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(ctx.stream);
  const auto num_gaussians = m_gaussians->size();
  const int original_num_gaussians = ctx.densification_info->size();
  const auto abs_ss_threshold = max(ctx.fwd_input.width, ctx.fwd_input.height) * m_params.max_screen_size;

  // Identify standard prune candidates (same criteria as default)
  thrust::device_vector<char> standard_prune(num_gaussians, 0);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());

  thrust::for_each(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_prune = standard_prune.data(), d_opacity,
       scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),
       scene_scale = m_gaussians->scene_scale(),
       rotation = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
       pruning_scale_threshold = m_params.pruning_scale_threshold,
       prune_large = this_step() > m_params.reset_every,
       max_radii_threshold = abs_ss_threshold,
       original_num_gaussians,
       deninfo = ctx.densification_info->data(),
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) {
        bool not_large_ws = max(activate_scale(scale[i])) < pruning_scale_threshold * scene_scale;
        bool not_large_ss = i >= original_num_gaussians ||
                            deninfo[i].max_radii_screen < max_radii_threshold;
        bool not_transparent = activate_opacity(d_opacity[i]) > min_opacity;
        bool not_degenerate = sum(abs(rotation[i])) > FLT_EPSILON;

        // Mark Gaussians that SHOULD be pruned = 1
        if (!(not_transparent && ((not_large_ws && not_large_ss) || !prune_large) && not_degenerate)) {
          d_prune[i] = 1;
        }
      });

  // Count standard candidates
  int num_standard_candidates = thrust::reduce(exec,
      standard_prune.begin(), standard_prune.end(), 0, thrust::plus<int>());

  if (num_standard_candidates == 0) {
    log_info("[FastGS] No prune candidates.");
    return;
  }

  // Budget = prune_budget_ratio * standard candidates
  const int budget = static_cast<int>(m_prune_budget_ratio * num_standard_candidates);
  if (budget <= 0) {
    log_info("[FastGS] Prune budget is 0.");
    return;
  }

  // Build is_alive flag — use budget-based selection if we have pruning scores,
  // otherwise fall back to pruning all candidates.
  thrust::device_vector<char> is_alive(num_gaussians, 1);
  const bool has_pruning_scores = (m_pruning_score.size() == num_gaussians);

  if (has_pruning_scores && budget < num_standard_candidates) {
    // Collect candidate indices
    thrust::device_vector<int> candidate_indices(num_standard_candidates);
    thrust::copy_if(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        standard_prune.begin(),
        candidate_indices.begin(),
        [] __device__(char p) { return p == 1; });

    // Compute prune weights = 1 / (1e-6 + 1 - pruning_score) for candidates
    // Higher pruning_score → higher weight → more likely to be pruned
    thrust::device_vector<float> prune_weights(num_standard_candidates);
    const float* d_ps = thrust::raw_pointer_cast(m_pruning_score.data());
    thrust::transform(exec,
        candidate_indices.begin(), candidate_indices.end(),
        prune_weights.begin(),
        [d_ps] __device__(int i) -> float {
          return 1.0f / (1e-6f + 1.0f - d_ps[i]);
        });

    // Sort candidates by weight (descending) and take top `budget`
    // This approximates multinomial sampling deterministically
    thrust::device_vector<int> sorted_indices(num_standard_candidates);
    thrust::sequence(exec, sorted_indices.begin(), sorted_indices.end());
    thrust::sort_by_key(exec, prune_weights.begin(), prune_weights.end(),
        sorted_indices.begin(), thrust::greater<float>());

    // Mark the top `budget` as dead
    const int actual_prune = std::min(budget, num_standard_candidates);
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(actual_prune),
        [d_is_alive = is_alive.data(),
         d_candidate = thrust::raw_pointer_cast(candidate_indices.data()),
         d_sorted = thrust::raw_pointer_cast(sorted_indices.data())] __device__(int i) {
          const int candidate_idx = d_sorted[i];  // index into candidate_indices
          const int gaussian_idx = d_candidate[candidate_idx];
          d_is_alive[gaussian_idx] = 0;
        });
  } else {
    // No pruning scores or budget >= candidates: prune all standard candidates
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        [d_is_alive = is_alive.data(), d_prune = standard_prune.data()] __device__(int i) {
          if (d_prune[i]) d_is_alive[i] = 0;
        });
  }

  int nums_kept = thrust::reduce(exec, is_alive.begin(), is_alive.end(), 0, thrust::plus<int>());
  log_info("[FastGS] Prune: removed {} (kept {}, budget was {})",
           num_gaussians - nums_kept, nums_kept, budget);
  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept);
}

// ---------------------------------------------------------------------------
// final_prune() — periodic hard pruning based on pruning_score and opacity
// ---------------------------------------------------------------------------

void FastGSStrategy::final_prune(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(ctx.stream);
  const auto num_gaussians = m_gaussians->size();
  const bool has_pruning = (m_pruning_score.size() == num_gaussians);

  thrust::device_vector<char> is_alive(num_gaussians, 1);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  const float* d_ps = has_pruning ? thrust::raw_pointer_cast(m_pruning_score.data()) : nullptr;
  const float score_thresh = m_final_prune_score_threshold;
  const float opacity_thresh = m_final_prune_opacity_threshold;

  thrust::for_each(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_is_alive = is_alive.data(), d_opacity, d_ps,
       score_thresh, opacity_thresh] __device__(int i) {
        const bool low_opacity = activate_opacity(d_opacity[i]) < opacity_thresh;
        const bool high_prune_score = (d_ps != nullptr && d_ps[i] > score_thresh);
        if (low_opacity || high_prune_score) {
          d_is_alive[i] = 0;
        }
      });

  int nums_kept = thrust::reduce(exec, is_alive.begin(), is_alive.end(), 0, thrust::plus<int>());
  log_info("[FastGS] Final prune: removed {} (kept {})",
           num_gaussians - nums_kept, nums_kept);
  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept);

  // Reset densification info
  ctx.densification_info = std::make_shared<GPUBuffer<DensificationInfo>>(nums_kept);
  ctx.densification_info->memset(0);
  m_importance_score.clear();
  m_pruning_score.clear();
}

// ---------------------------------------------------------------------------
// set_params / get_params
// ---------------------------------------------------------------------------

void FastGSStrategy::set_params(const json& config) {
  StrategyBase::set_params(config);
  m_rng.seed(m_params.seed);
  if (config.contains("absgrad_threshold"))      m_absgrad_threshold = config["absgrad_threshold"].get<float>();
  if (config.contains("percent_dense"))          m_percent_dense = config["percent_dense"].get<float>();
  if (config.contains("loss_thresh"))            m_loss_thresh = config["loss_thresh"].get<float>();
  if (config.contains("metric_num_cameras"))     m_metric_num_cameras = config["metric_num_cameras"].get<int>();
  if (config.contains("importance_threshold"))   m_importance_threshold = config["importance_threshold"].get<float>();
  if (config.contains("prune_budget_ratio"))     m_prune_budget_ratio = config["prune_budget_ratio"].get<float>();
  if (config.contains("final_prune_score_threshold"))
    m_final_prune_score_threshold = config["final_prune_score_threshold"].get<float>();
  if (config.contains("final_prune_opacity_threshold"))
    m_final_prune_opacity_threshold = config["final_prune_opacity_threshold"].get<float>();
  if (config.contains("final_prune_start"))      m_final_prune_start = config["final_prune_start"].get<int>();
  if (config.contains("final_prune_end"))        m_final_prune_end = config["final_prune_end"].get<int>();
  if (config.contains("final_prune_every"))      m_final_prune_every = config["final_prune_every"].get<int>();
  if (config.contains("opacity_reset_value"))    m_opacity_reset_value = config["opacity_reset_value"].get<float>();
}

json FastGSStrategy::get_params() const {
  json params = StrategyBase::get_params();
  params["type"] = "fastgs";
  params["absgrad_threshold"] = m_absgrad_threshold;
  params["percent_dense"] = m_percent_dense;
  params["loss_thresh"] = m_loss_thresh;
  params["metric_num_cameras"] = m_metric_num_cameras;
  params["importance_threshold"] = m_importance_threshold;
  params["prune_budget_ratio"] = m_prune_budget_ratio;
  params["final_prune_score_threshold"] = m_final_prune_score_threshold;
  params["final_prune_opacity_threshold"] = m_final_prune_opacity_threshold;
  params["final_prune_start"] = m_final_prune_start;
  params["final_prune_end"] = m_final_prune_end;
  params["final_prune_every"] = m_final_prune_every;
  params["opacity_reset_value"] = m_opacity_reset_value;
  return params;
}

}  // namespace tinygs
