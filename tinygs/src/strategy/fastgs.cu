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
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <nvtx3/nvtx3.hpp>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "tinygs/cuda/common_device.cuh"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/loss/fused_ssim.hpp"
#include "tinygs/loss/l1.hpp"
#include "random/device.cuh"
#include "tinygs/random/multinomial.hpp"
#include "tinygs/strategy/fastgs.hpp"
#include "tinygs/utils/image_format.hpp"

namespace tinygs {

namespace {

struct NumericMoments {
  long long count = 0;
  long long finite_count = 0;
  long long non_finite_count = 0;
  double sum = 0.0;
  double sum_sq = 0.0;
  float min_value = FLT_MAX;
  float max_value = -FLT_MAX;
};

struct NumericMomentsAdd {
  __host__ __device__ NumericMoments operator()(
      const NumericMoments& a,
      const NumericMoments& b) const {
    NumericMoments out;
    out.count = a.count + b.count;
    out.finite_count = a.finite_count + b.finite_count;
    out.non_finite_count = a.non_finite_count + b.non_finite_count;
    out.sum = a.sum + b.sum;
    out.sum_sq = a.sum_sq + b.sum_sq;
    out.min_value = fminf(a.min_value, b.min_value);
    out.max_value = fmaxf(a.max_value, b.max_value);
    return out;
  }
};

struct DuplicateVerboseStats {
  long long total = 0;
  long long counter_non_positive = 0;
  long long importance_unavailable = 0;
  long long importance_pass = 0;
  long long importance_fail = 0;
  long long grad_non_finite_before_sanitize = 0;
  long long absgrad_non_finite_before_sanitize = 0;
  long long small_scale = 0;
  long long large_scale = 0;
  long long grad_pass = 0;
  long long absgrad_pass = 0;
  long long clone = 0;
  long long split = 0;
  long long reject_importance = 0;
  long long reject_small_grad = 0;
  long long reject_large_absgrad = 0;
};

struct DuplicateVerboseStatsAdd {
  __host__ __device__ DuplicateVerboseStats operator()(
      const DuplicateVerboseStats& a,
      const DuplicateVerboseStats& b) const {
    DuplicateVerboseStats out;
    out.total = a.total + b.total;
    out.counter_non_positive = a.counter_non_positive + b.counter_non_positive;
    out.importance_unavailable = a.importance_unavailable + b.importance_unavailable;
    out.importance_pass = a.importance_pass + b.importance_pass;
    out.importance_fail = a.importance_fail + b.importance_fail;
    out.grad_non_finite_before_sanitize =
        a.grad_non_finite_before_sanitize + b.grad_non_finite_before_sanitize;
    out.absgrad_non_finite_before_sanitize =
        a.absgrad_non_finite_before_sanitize + b.absgrad_non_finite_before_sanitize;
    out.small_scale = a.small_scale + b.small_scale;
    out.large_scale = a.large_scale + b.large_scale;
    out.grad_pass = a.grad_pass + b.grad_pass;
    out.absgrad_pass = a.absgrad_pass + b.absgrad_pass;
    out.clone = a.clone + b.clone;
    out.split = a.split + b.split;
    out.reject_importance = a.reject_importance + b.reject_importance;
    out.reject_small_grad = a.reject_small_grad + b.reject_small_grad;
    out.reject_large_absgrad = a.reject_large_absgrad + b.reject_large_absgrad;
    return out;
  }
};

struct PruneReasonStats {
  long long selected = 0;
  long long low_opacity = 0;
  long long large_world = 0;
  long long large_screen = 0;
  long long degenerate_rotation = 0;
  long long only_low_opacity = 0;
  long long only_large_world = 0;
  long long only_large_screen = 0;
  long long only_degenerate_rotation = 0;
  long long multi_reason = 0;
};

struct PruneReasonStatsAdd {
  __host__ __device__ PruneReasonStats operator()(
      const PruneReasonStats& a,
      const PruneReasonStats& b) const {
    PruneReasonStats out;
    out.selected = a.selected + b.selected;
    out.low_opacity = a.low_opacity + b.low_opacity;
    out.large_world = a.large_world + b.large_world;
    out.large_screen = a.large_screen + b.large_screen;
    out.degenerate_rotation = a.degenerate_rotation + b.degenerate_rotation;
    out.only_low_opacity = a.only_low_opacity + b.only_low_opacity;
    out.only_large_world = a.only_large_world + b.only_large_world;
    out.only_large_screen = a.only_large_screen + b.only_large_screen;
    out.only_degenerate_rotation = a.only_degenerate_rotation + b.only_degenerate_rotation;
    out.multi_reason = a.multi_reason + b.multi_reason;
    return out;
  }
};

struct FinalPruneReasonStats {
  long long removed = 0;
  long long low_opacity = 0;
  long long high_pruning_score = 0;
  long long only_low_opacity = 0;
  long long only_high_pruning_score = 0;
  long long both_low_and_high = 0;
};

struct FinalPruneReasonStatsAdd {
  __host__ __device__ FinalPruneReasonStats operator()(
      const FinalPruneReasonStats& a,
      const FinalPruneReasonStats& b) const {
    FinalPruneReasonStats out;
    out.removed = a.removed + b.removed;
    out.low_opacity = a.low_opacity + b.low_opacity;
    out.high_pruning_score = a.high_pruning_score + b.high_pruning_score;
    out.only_low_opacity = a.only_low_opacity + b.only_low_opacity;
    out.only_high_pruning_score = a.only_high_pruning_score + b.only_high_pruning_score;
    out.both_low_and_high = a.both_low_and_high + b.both_low_and_high;
    return out;
  }
};

NumericMoments reduce_numeric_moments(
    cudaStream_t stream,
    const float* values,
    int n,
    const char* mask = nullptr,
    bool include_mask_nonzero = true) {
  auto exec = thrust::cuda::par.on(stream);
  const NumericMoments zero{};
  return thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(n),
      [values, mask, include_mask_nonzero] __device__(int i) -> NumericMoments {
        NumericMoments out{};
        if (mask != nullptr) {
          const bool mask_selected = mask[i] != 0;
          if (mask_selected != include_mask_nonzero) return out;
        }
        out.count = 1;
        const float value = values[i];
        if (isfinite(value)) {
          out.finite_count = 1;
          out.sum = static_cast<double>(value);
          out.sum_sq = static_cast<double>(value) * static_cast<double>(value);
          out.min_value = value;
          out.max_value = value;
        } else {
          out.non_finite_count = 1;
        }
        return out;
      },
      zero,
      NumericMomentsAdd{});
}

long long count_above_threshold(
    cudaStream_t stream,
    const float* values,
    int n,
    float threshold,
    const char* mask = nullptr,
    bool include_mask_nonzero = true) {
  auto exec = thrust::cuda::par.on(stream);
  return thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(n),
      [values, threshold, mask, include_mask_nonzero] __device__(int i) -> long long {
        if (mask != nullptr) {
          const bool mask_selected = mask[i] != 0;
          if (mask_selected != include_mask_nonzero) return 0;
        }
        const float value = values[i];
        return (isfinite(value) && value > threshold) ? 1 : 0;
      },
      0ll,
      thrust::plus<long long>());
}

void log_numeric_moments(const std::string& label, const NumericMoments& stats) {
  if (stats.count == 0) {
    printf("[FastGS][Verbose] %s: count=0\n", label.c_str());
    return;
  }
  if (stats.finite_count == 0) {
    printf("[FastGS][Verbose] %s: count=%lld finite=0 non_finite=%lld\n",
           label.c_str(), stats.count, stats.non_finite_count);
    return;
  }
  const double mean = stats.sum / static_cast<double>(stats.finite_count);
  const double var = std::max(0.0, stats.sum_sq / static_cast<double>(stats.finite_count) - mean * mean);
  const double stddev = std::sqrt(var);
  printf(
      "[FastGS][Verbose] %s: count=%lld finite=%lld non_finite=%lld min=%.6g max=%.6g mean=%.6g std=%.6g\n",
      label.c_str(),
      stats.count,
      stats.finite_count,
      stats.non_finite_count,
      stats.min_value,
      stats.max_value,
      mean,
      stddev);
}

void log_prune_reason_stats(const std::string& label, const PruneReasonStats& stats) {
  printf(
      "[FastGS][Verbose] %s: selected=%lld | low_opacity=%lld large_ws=%lld large_ss=%lld degenerate=%lld | "
      "only(low/ws/ss/deg)=%lld/%lld/%lld/%lld multi_reason=%lld\n",
      label.c_str(),
      stats.selected,
      stats.low_opacity,
      stats.large_world,
      stats.large_screen,
      stats.degenerate_rotation,
      stats.only_low_opacity,
      stats.only_large_world,
      stats.only_large_screen,
      stats.only_degenerate_rotation,
      stats.multi_reason);
}

void log_final_prune_reason_stats(const FinalPruneReasonStats& stats) {
  printf(
      "[FastGS][Verbose] final_prune.reasons: removed=%lld low_opacity=%lld high_pruning_score=%lld "
      "only_low=%lld only_high_score=%lld both=%lld\n",
      stats.removed,
      stats.low_opacity,
      stats.high_pruning_score,
      stats.only_low_opacity,
      stats.only_high_pruning_score,
      stats.both_low_and_high);
}

}  // namespace

struct FastGSStrategy::Impl {
  thrust::device_vector<float> m_importance_score;
  thrust::device_vector<float> m_pruning_score;
};

#define m_importance_score m_impl->m_importance_score
#define m_pruning_score m_impl->m_pruning_score

FastGSStrategy::FastGSStrategy(
    std::shared_ptr<BackendRuntime> runtime,
    std::shared_ptr<GPUGaussian3d> gaussians,
    std::shared_ptr<GPUGaussian3d> gaussians_grad,
    std::shared_ptr<OptimizerBase> optimizer)
    : StrategyBase(runtime, gaussians, gaussians_grad, optimizer), m_impl(std::make_unique<Impl>()) {
  // Reference FastGS training defaults (ref_impl/FastGS/train_base.sh)
  // use densification every 500 iterations.
  m_params.refine_every = 500;
}

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

/// @brief Compute per-pixel mean L1 loss across 3 channels.
///        Supports Float32 (CHW tiled 8x8) images.
__global__ void compute_l1_map_kernel(
    int n_pixels,
    const float* __restrict__ rendered,    // CHW tiled image (rendered)
    const float* __restrict__ gt,          // CHW tiled image (ground truth)
    float* __restrict__ l1_map,            // H*W output mean-L1 map (flat)
    uint32_t tiled_w,                      // padded_width / 8 (tiles per row)
    uint32_t padded_w,                     // padded width (pixels per row per channel)
    uint32_t padded_h,                     // padded height
    uint32_t width,                        // actual width
    uint32_t height) {                     // actual height
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pixels) return;
  const uint32_t py = idx / width;
  const uint32_t px = idx % width;
  if (py >= height) return;

  // Tiled linear index for 8x8 tile storage
  const uint32_t linear_idx = get_linear_index_tiled(py, px, tiled_w);
  const uint32_t channel_stride = padded_w * padded_h;

  // Accumulate L1 across 3 channels in CHW tiled layout
  float l1_sum = 0.0f;
  for (int c = 0; c < 3; c++) {
    const uint32_t offset = c * channel_stride + linear_idx;
    l1_sum += fabsf(fminf(fmaxf(rendered[offset], 0.0f), 1.0f) - gt[offset]);
  }
  l1_map[idx] = l1_sum / 3.0f;
}

/// @brief Threshold per-pixel L1 map to a binary metric map.
///        Optionally min-max normalizes L1 values to [0,1] before thresholding.
__global__ void threshold_metric_map_kernel(
    int n_pixels,
    const float* __restrict__ l1_map,
    int* __restrict__ metric_map,
    float loss_thresh,
    float min_l1,
    float inv_range,
    bool normalize_l1) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pixels) return;
  float value = l1_map[idx];
  if (normalize_l1) {
    value = (value - min_l1) * inv_range;
    value = fminf(fmaxf(value, 0.0f), 1.0f);
  }
  metric_map[idx] = (value > loss_thresh) ? 1 : 0;
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
//
// Exact implementation of compute_gaussian_score_fastgs from the reference:
//   ref_impl/FastGS/utils/fast_utils.py
//
// For each camera:
//   1. Render the scene (first render) to get the rendered image.
//   2. Compute per-pixel mean L1 loss vs GT, min-max normalize to [0,1], then threshold
//      to get a binary metric_map.
//   3. Compute photometric loss = l1_weight*L1 + ssim_weight*(1-SSIM)
//   4. Render again (second render) with metric_mode=true and metric_map set,
//      to get per-Gaussian accum_metric_counts via atomicAdd in the blend kernel.
//   5. Accumulate full_metric_counts += accum_metric_counts  (if densify=true)
//   6. Accumulate full_metric_score += photometric_loss * accum_metric_counts
//
// After all cameras:
//   - pruning_score = min-max normalize(full_metric_score)
//   - importance_score = floor(full_metric_counts / num_cameras)  (if densify=true)
// ---------------------------------------------------------------------------

void FastGSStrategy::compute_gaussian_score(const RasterizeContext& ctx, bool densify) {
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

  // Allocate per-Gaussian cumulative accumulators
  thrust::device_vector<float> full_metric_score(num_gaussians, 0.0f);
  thrust::device_vector<int> full_metric_counts(num_gaussians, 0);

  // Match the temporary render dtype to the active train output dtype.
  const DataType render_dtype = ctx.fwd_output.image.data_type;
  if (render_dtype != DataType::Float16 && render_dtype != DataType::Float32) {
    throw std::runtime_error("FastGS metric scoring expects Float16 or Float32 render dtype.");
  }

  // IMPORTANT: use the active training resolution from ctx (progressive resolution aware),
  // not the raw dataset image size.
  const int width = static_cast<int>(ctx.fwd_input.width);
  const int height = static_cast<int>(ctx.fwd_input.height);
  const int n_pixels = width * height;
  ImageShape rgb_shape{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3};
  const int padded_w = static_cast<int>(rgb_shape.padded_width());
  const int padded_h = static_cast<int>(rgb_shape.padded_height());
  const uint32_t tiled_w = rgb_shape.tiled_width();
  const size_t rgb_padded_size = static_cast<size_t>(rgb_shape.padded_size());

  // Reuse scratch buffers across all sampled cameras.
  thrust::device_vector<float> render_buf_f32;
  thrust::device_vector<float16_t> render_buf_f16;
  thrust::device_vector<float> metric_render_buf_f32;
  thrust::device_vector<float16_t> metric_render_buf_f16;
  if (render_dtype == DataType::Float16) {
    render_buf_f16.resize(rgb_padded_size);
    metric_render_buf_f16.resize(rgb_padded_size);
  } else {
    render_buf_f32.resize(rgb_padded_size);
    metric_render_buf_f32.resize(rgb_padded_size);
  }

  thrust::device_vector<float> gt_gpu(rgb_padded_size);
  thrust::device_vector<float> rendered_f32(rgb_padded_size);
  thrust::device_vector<float> l1_loss_buf(rgb_padded_size);
  thrust::device_vector<float> ssim_loss_buf(rgb_padded_size);
  thrust::device_vector<float> l1_map(n_pixels);
  auto metric_map = create_device_buffer_for<int>(ctx.runtime, n_pixels);
  auto metric_counts = create_device_buffer_for<int>(ctx.runtime, num_gaussians);
  Image gt_image(rgb_shape, DataType::Float32, thrust::raw_pointer_cast(gt_gpu.data()));

  // Shared loss contexts reused for every camera.
  L1Loss l1_loss(ctx.runtime);
  FusedSSIMLoss ssim_loss(ctx.runtime);
  LossContext l1_ctx;
  l1_ctx.pred = Image(rgb_shape, DataType::Float32, thrust::raw_pointer_cast(rendered_f32.data()));
  l1_ctx.target = gt_image;
  l1_ctx.loss = Image(rgb_shape, DataType::Float32, thrust::raw_pointer_cast(l1_loss_buf.data()));
  l1_ctx.grad = Image();
  l1_ctx.queue = ctx.queue;

  LossContext ssim_ctx;
  ssim_ctx.pred = Image(rgb_shape, DataType::Float32, thrust::raw_pointer_cast(rendered_f32.data()));
  ssim_ctx.target = gt_image;
  ssim_ctx.loss = Image(rgb_shape, DataType::Float32, thrust::raw_pointer_cast(ssim_loss_buf.data()));
  ssim_ctx.grad = Image();
  ssim_ctx.queue = ctx.queue;

  // Number of cameras to render
  const int num_cameras = std::min(m_metric_num_cameras, static_cast<int>(dataset_size));
  if (m_print_verbose_stats) {
    printf(
        "[FastGS][Verbose] compute_gaussian_score(densify=%d): gaussians=%zu dataset=%zu sampled_cameras=%d "
        "loss_thresh=%.6g normalize_metric_l1=%d photometric(l1,ssim)=(%.6g, %.6g)\n",
        densify,
        num_gaussians,
        dataset_size,
        num_cameras,
        m_loss_thresh,
        m_normalize_metric_l1,
        m_photometric_l1_weight,
        m_photometric_ssim_weight);
  }

  // Build sampled camera indices.
  std::vector<size_t> sampled_indices(static_cast<size_t>(num_cameras), 0);
  if (m_sample_cameras_without_replacement) {
    std::vector<size_t> all_indices(dataset_size);
    std::iota(all_indices.begin(), all_indices.end(), size_t{0});
    for (int i = 0; i < num_cameras; ++i) {
      const size_t remaining = dataset_size - static_cast<size_t>(i);
      const size_t j = static_cast<size_t>(i) +
          static_cast<size_t>(m_rng.next_uint(static_cast<uint32_t>(remaining)));
      std::swap(all_indices[static_cast<size_t>(i)], all_indices[j]);
      sampled_indices[static_cast<size_t>(i)] = all_indices[static_cast<size_t>(i)];
    }
  } else {
    for (int i = 0; i < num_cameras; ++i) {
      sampled_indices[static_cast<size_t>(i)] =
          static_cast<size_t>(m_rng.next_uint(static_cast<uint32_t>(dataset_size)));
    }
  }

  for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
    // Pick a sampled camera
    const size_t idx = sampled_indices[static_cast<size_t>(cam_i)];
    auto data = (*dataset)[idx];

    // -- First render: get the rendered image (no metric counting) --
    RasterizeContext render_ctx;
    render_ctx.inference = true;
    render_ctx.runtime = ctx.runtime;
    render_ctx.queue = ctx.queue;
    render_ctx.gaussians_grad = ctx.gaussians_grad;
    render_ctx.grad_scaler = ctx.grad_scaler;
    render_ctx.metric_mode = false;

    render_ctx.fwd_input.width = width;
    render_ctx.fwd_input.height = height;
    render_ctx.fwd_input.K = data.K;
    render_ctx.fwd_input.w2c = data.w2c;
    render_ctx.fwd_input.near = ctx.fwd_input.near;
    render_ctx.fwd_input.far = ctx.fwd_input.far;
    render_ctx.fwd_input.timestamp = data.timestamp;

    if (render_dtype == DataType::Float16) {
      thrust::fill(render_buf_f16.begin(), render_buf_f16.end(), float16_t(0));
      render_ctx.fwd_output.image = Image(rgb_shape, DataType::Float16, thrust::raw_pointer_cast(render_buf_f16.data()));
    } else {
      thrust::fill(render_buf_f32.begin(), render_buf_f32.end(), 0.0f);
      render_ctx.fwd_output.image = Image(rgb_shape, DataType::Float32, thrust::raw_pointer_cast(render_buf_f32.data()));
    }

    m_rasterizer->forward(render_ctx);
    CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue)));

    // Transfer GT to GPU
    m_dataloader->transfer_gpu(ctx.queue.get(), gt_image, data.image);
    CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue)));

    // Convert rendered image to Float32 if needed, then compute metrics in Float32.
    if (render_dtype == DataType::Float16) {
      half_to_float_gpu(thrust::raw_pointer_cast(rendered_f32.data()),
        reinterpret_cast<const float16_t*>(render_ctx.fwd_output.image.data),
        rgb_padded_size,
        ctx.queue.get());
    } else {
      CUDA_CHECK_THROW(cudaMemcpyAsync(thrust::raw_pointer_cast(rendered_f32.data()), render_ctx.fwd_output.image.data,
        rgb_padded_size * sizeof(float), cudaMemcpyDeviceToDevice, to_cuda_stream(ctx.queue)));
    }

    // Compute per-pixel L1 map and threshold into binary metric map.
    linear_kernel(compute_l1_map_kernel, 0, to_cuda_stream(ctx.queue), n_pixels,
        thrust::raw_pointer_cast(rendered_f32.data()),
        static_cast<const float*>(gt_image.data),
        thrust::raw_pointer_cast(l1_map.data()),
        tiled_w,
        static_cast<uint32_t>(padded_w),
        static_cast<uint32_t>(padded_h),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height));

    auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
    auto l1_begin = thrust::device_pointer_cast(thrust::raw_pointer_cast(l1_map.data()));
    auto l1_end = l1_begin + n_pixels;
    float min_l1 = thrust::reduce(exec, l1_begin, l1_end,
        std::numeric_limits<float>::max(), thrust::minimum<float>());
    float max_l1 = thrust::reduce(exec, l1_begin, l1_end,
        std::numeric_limits<float>::lowest(), thrust::maximum<float>());
    float range = max_l1 - min_l1;
    if (range < 1e-8f) {
      range = 1.0f;
      min_l1 = 0.0f;
    }
    const float inv_range = 1.0f / range;

    linear_kernel(threshold_metric_map_kernel, 0, to_cuda_stream(ctx.queue), n_pixels,
        thrust::raw_pointer_cast(l1_map.data()),
        buffer_data<int>(metric_map),
        m_loss_thresh,
        min_l1,
        inv_range,
        m_normalize_metric_l1);

    // Compute scalar photometric loss:
    //   l1_weight * mean(L1) + ssim_weight * mean(1 - SSIM)
    float photometric_loss_h = 0.0f;
    {
      thrust::fill(l1_loss_buf.begin(), l1_loss_buf.end(), 0.0f);
      thrust::fill(ssim_loss_buf.begin(), ssim_loss_buf.end(), 0.0f);

      l1_loss.evaluate(l1_ctx, 1.0f);

      ssim_loss.evaluate(ssim_ctx, 1.0f);

      auto l1_loss_begin = thrust::device_pointer_cast(thrust::raw_pointer_cast(l1_loss_buf.data()));
      auto l1_loss_end = l1_loss_begin + static_cast<int>(rgb_padded_size);
      auto ssim_loss_begin = thrust::device_pointer_cast(thrust::raw_pointer_cast(ssim_loss_buf.data()));
      auto ssim_loss_end = ssim_loss_begin + static_cast<int>(rgb_padded_size);
      const float l1_term = thrust::reduce(exec, l1_loss_begin, l1_loss_end, 0.0f, thrust::plus<float>());
      const float ssim_term = thrust::reduce(exec, ssim_loss_begin, ssim_loss_end, 0.0f, thrust::plus<float>());
      photometric_loss_h = m_photometric_l1_weight * l1_term +
                           m_photometric_ssim_weight * ssim_term;
    }

    // -- Second render: with metric_mode=true to count per-Gaussian contributions --
    RasterizeContext metric_ctx;
    metric_ctx.inference = true;
    metric_ctx.runtime = ctx.runtime;
    metric_ctx.queue = ctx.queue;
    metric_ctx.gaussians_grad = ctx.gaussians_grad;
    metric_ctx.grad_scaler = ctx.grad_scaler;
    metric_ctx.metric_mode = true;

    metric_ctx.fwd_input = render_ctx.fwd_input;  // same camera

    if (render_dtype == DataType::Float16) {
      thrust::fill(metric_render_buf_f16.begin(), metric_render_buf_f16.end(), float16_t(0));
      metric_ctx.fwd_output.image = Image(rgb_shape, DataType::Float16, thrust::raw_pointer_cast(metric_render_buf_f16.data()));
    } else {
      thrust::fill(metric_render_buf_f32.begin(), metric_render_buf_f32.end(), 0.0f);
      metric_ctx.fwd_output.image = Image(rgb_shape, DataType::Float32, thrust::raw_pointer_cast(metric_render_buf_f32.data()));
    }

    // Set metric_map and metric_counts
    metric_ctx.metric_map = metric_map;
    metric_ctx.metric_counts = metric_counts;
    fill_buffer_zero_async(ctx.runtime, ctx.queue, metric_ctx.metric_counts);

    m_rasterizer->forward_metric(metric_ctx);
    CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue)));

    // Accumulate results
    exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
    const int* d_accum_counts = buffer_data<int>(metric_counts);
    float* d_full_score = thrust::raw_pointer_cast(full_metric_score.data());
    int* d_full_counts = thrust::raw_pointer_cast(full_metric_counts.data());
    const float ploss = photometric_loss_h;

    if (densify) {
      // full_metric_counts += accum_loss_counts
      thrust::transform(exec,
          full_metric_counts.begin(), full_metric_counts.end(),
          thrust::device_pointer_cast(d_accum_counts),
          full_metric_counts.begin(),
          thrust::plus<int>());
    }

    if (m_print_verbose_stats) {
      const long long metric_pixels = thrust::transform_reduce(
          exec,
          thrust::make_counting_iterator<int>(0),
          thrust::make_counting_iterator<int>(n_pixels),
          [d_metric_map = buffer_data<int>(metric_map)] __device__(int i) -> long long {
            return d_metric_map[i] > 0 ? 1 : 0;
          },
          0ll,
          thrust::plus<long long>());
      const long long nonzero_metric_gaussians = thrust::transform_reduce(
          exec,
          thrust::make_counting_iterator<int>(0),
          thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
          [d_accum_counts] __device__(int i) -> long long {
            return d_accum_counts[i] > 0 ? 1 : 0;
          },
          0ll,
          thrust::plus<long long>());
      const long long metric_count_sum = thrust::transform_reduce(
          exec,
          thrust::make_counting_iterator<int>(0),
          thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
          [d_accum_counts] __device__(int i) -> long long {
            return static_cast<long long>(d_accum_counts[i]);
          },
          0ll,
          thrust::plus<long long>());
      const int max_metric_count = thrust::transform_reduce(
          exec,
          thrust::make_counting_iterator<int>(0),
          thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
          [d_accum_counts] __device__(int i) -> int { return d_accum_counts[i]; },
          0,
          thrust::maximum<int>());
      const double metric_pixel_ratio =
          n_pixels > 0 ? static_cast<double>(metric_pixels) / static_cast<double>(n_pixels) : 0.0;
      printf(
          "[FastGS][Verbose] Score camera %d/%d (dataset idx=%zu): "
          "l1[min,max]=(%.6g,%.6g) metric_pixels=%lld (%.2f%%) photometric_loss=%.6g "
          "gaussians_hit=%lld metric_count_sum=%lld max_metric_count=%d\n",
          cam_i + 1,
          num_cameras,
          idx,
          min_l1,
          max_l1,
          metric_pixels,
          metric_pixel_ratio * 100.0,
          photometric_loss_h,
          nonzero_metric_gaussians,
          metric_count_sum,
          max_metric_count);
    }

    // full_metric_score += photometric_loss * accum_loss_counts
    const int ng = static_cast<int>(num_gaussians);
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(ng),
        [d_full_score, d_accum_counts, ploss] __device__(int i) {
          d_full_score[i] += ploss * static_cast<float>(d_accum_counts[i]);
        });
  }

  // -- Compute final scores --

  // pruning_score = min-max normalize(full_metric_score) to [0, 1]
  m_pruning_score.resize(num_gaussians);
  {
    auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
    float min_score = thrust::reduce(exec, full_metric_score.begin(), full_metric_score.end(),
        std::numeric_limits<float>::max(), thrust::minimum<float>());
    float max_score = thrust::reduce(exec, full_metric_score.begin(), full_metric_score.end(),
        std::numeric_limits<float>::lowest(), thrust::maximum<float>());
    float range = max_score - min_score;
    if (range < 1e-8f) range = 1.0f;  // avoid division by zero

    thrust::transform(exec,
        full_metric_score.begin(), full_metric_score.end(),
        m_pruning_score.begin(),
        [min_score, range] __device__(float score) -> float {
          return (score - min_score) / range;
        });
  }

  // importance_score = floor(full_metric_counts / num_cameras) if densify
  if (densify) {
    m_importance_score.resize(num_gaussians);
    auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
    const float inv_cams = 1.0f / static_cast<float>(num_cameras);
    thrust::transform(exec,
        full_metric_counts.begin(), full_metric_counts.end(),
        m_importance_score.begin(),
        [inv_cams] __device__(int count) -> float {
          return floorf(static_cast<float>(count) * inv_cams);
        });
  }

  if (m_print_verbose_stats) {
    auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
    const NumericMoments raw_metric_score_stats = reduce_numeric_moments(
        to_cuda_stream(ctx.queue),
        thrust::raw_pointer_cast(full_metric_score.data()),
        static_cast<int>(num_gaussians));
    log_numeric_moments("score.raw_metric_score", raw_metric_score_stats);

    const long long active_metric_counts = thrust::transform_reduce(
        exec,
        full_metric_counts.begin(),
        full_metric_counts.end(),
        [] __device__(int c) -> long long { return c > 0 ? 1 : 0; },
        0ll,
        thrust::plus<long long>());
    const long long metric_count_sum = thrust::transform_reduce(
        exec,
        full_metric_counts.begin(),
        full_metric_counts.end(),
        [] __device__(int c) -> long long { return static_cast<long long>(c); },
        0ll,
        thrust::plus<long long>());
    int min_metric_count = 0;
    int max_metric_count = 0;
    if (num_gaussians > 0) {
      min_metric_count = thrust::reduce(
          exec,
          full_metric_counts.begin(),
          full_metric_counts.end(),
          std::numeric_limits<int>::max(),
          thrust::minimum<int>());
      max_metric_count = thrust::reduce(
          exec,
          full_metric_counts.begin(),
          full_metric_counts.end(),
          std::numeric_limits<int>::lowest(),
          thrust::maximum<int>());
    }
    const double mean_metric_count = num_gaussians > 0
        ? static_cast<double>(metric_count_sum) / static_cast<double>(num_gaussians)
        : 0.0;
    printf(
        "[FastGS][Verbose] score.metric_counts: sum=%lld active=%lld min=%d max=%d mean=%.6g\n",
        metric_count_sum,
        active_metric_counts,
        min_metric_count,
        max_metric_count,
        mean_metric_count);

    const float* d_pruning = thrust::raw_pointer_cast(m_pruning_score.data());
    const NumericMoments pruning_stats = reduce_numeric_moments(
        to_cuda_stream(ctx.queue),
        d_pruning,
        static_cast<int>(num_gaussians));
    const long long pruning_gt_05 = count_above_threshold(
        to_cuda_stream(ctx.queue),
        d_pruning,
        static_cast<int>(num_gaussians),
        0.5f);
    const long long pruning_gt_09 = count_above_threshold(
        to_cuda_stream(ctx.queue),
        d_pruning,
        static_cast<int>(num_gaussians),
        0.9f);
    log_numeric_moments("score.pruning_score", pruning_stats);
    printf(
        "[FastGS][Verbose] score.pruning_score threshold counts: >0.5=%lld >0.9=%lld\n",
        pruning_gt_05,
        pruning_gt_09);

    if (densify) {
      const float* d_importance = thrust::raw_pointer_cast(m_importance_score.data());
      const NumericMoments importance_stats = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_importance,
          static_cast<int>(num_gaussians));
      const long long importance_gt_thresh = count_above_threshold(
          to_cuda_stream(ctx.queue),
          d_importance,
          static_cast<int>(num_gaussians),
          m_importance_threshold);
      log_numeric_moments("score.importance_score", importance_stats);
      printf(
          "[FastGS][Verbose] score.importance_score threshold count: >%.6g = %lld\n",
          m_importance_threshold,
          importance_gt_thresh);
    }
  }

  CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue)));
}

// ---------------------------------------------------------------------------
// step_impl()
// ---------------------------------------------------------------------------

void FastGSStrategy::step_impl(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  CUDA_CHECK_THROW(cudaStreamSynchronize(to_cuda_stream(ctx.queue)));

  if (!ctx.densification_info) {
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(ctx.runtime, num_gaussians);
    fill_buffer_zero_async(ctx.runtime, ctx.queue, ctx.densification_info);
  }

  const int step = this_step();
    if (step % m_params.refine_every == 0 &&
      step >= m_params.start_refine &&
      step < m_params.end_refine) {

    // Compute multi-view importance scores before densification
    compute_gaussian_score(ctx, /*densify=*/ true);

    if (m_gaussians->size() < m_params.max_num_gaussians) {
      duplicate(ctx);
    }

    // Pad scores for newly added Gaussians so budget-based pruning can operate.
    // Reference: new Gaussians get padded_importance=0, protecting them from
    // multinomial sampling. Here pruning_score=0 yields weight~1 which is
    // negligible vs high-score Gaussians (weight>>100).
    const size_t new_size = m_gaussians->size();
    if (!m_pruning_score.empty() && m_pruning_score.size() < new_size) {
      m_pruning_score.resize(new_size, 0.0f);
    }
    if (!m_importance_score.empty() && m_importance_score.size() < new_size) {
      m_importance_score.resize(new_size, 0.0f);
    }

    prune(ctx);

    // Clamp opacity after densification (FastGS specific)
    linear_kernel(clamp_opacity_kernel, 0, to_cuda_stream(ctx.queue),
        static_cast<int>(m_gaussians->size()),
        thrust::raw_pointer_cast(m_gaussians->opacities().data()),
        0.8);

    // Reset opacity Adam state after clamping (reference: replace_tensor_to_optimizer zeros exp_avg/exp_avg_sq)
    on_reset_opacity(ctx.queue);

    // Reset densification info since indices have changed
    size_t num_gaussians = m_gaussians->size();
    ctx.densification_info = create_device_buffer_for<DensificationInfo>(ctx.runtime, num_gaussians);
    fill_buffer_zero_async(ctx.runtime, ctx.queue, ctx.densification_info);
    m_importance_score.clear();
    m_pruning_score.clear();
  }

  // Final prune: remove Gaussians with high pruning_score or low opacity
  if (step >= m_final_prune_start && step <= m_final_prune_end &&
      step % m_final_prune_every == 0) {
    // Need to recompute scores for final prune
    if (m_importance_score.empty()) {
      compute_gaussian_score(ctx, /*densify=*/ false);
    }
    final_prune(ctx);
  }

  if (m_params.reset_every > 0 && step % m_params.reset_every == 0 &&
      step > m_params.start_refine && step < m_params.end_refine) {
    linear_kernel(clamp_opacity_kernel, 0, to_cuda_stream(ctx.queue),
        static_cast<int>(m_gaussians->size()),
        thrust::raw_pointer_cast(m_gaussians->opacities().data()),
        m_opacity_reset_value);
    on_reset_opacity(ctx.queue);
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
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
  const auto num_gaussians = m_gaussians->size();

  if (!ctx.densification_info || buffer_count<DensificationInfo>(ctx.densification_info) != num_gaussians) {
    log_warning("[FastGS] Densification info is not provided or has wrong size, skip duplication.");
    return;
  }

  const bool has_importance = (m_importance_score.size() == num_gaussians);

  // Per-Gaussian flag: 0 = nothing, 1 = clone, 2 = split
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
  const float importance_thresh = m_importance_threshold;
  const float* d_importance = has_importance ?
      thrust::raw_pointer_cast(m_importance_score.data()) : nullptr;

  // Classify each Gaussian as clone, split, or nothing
  thrust::for_each(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_densification_info, d_scale, d_grow_flags, d_importance,
       clone_thresh, split_thresh, scale_boundary, importance_thresh,
       sanitize_non_finite = m_sanitize_nan_gradients] __device__(int i) {
        const float counter = fmaxf(d_densification_info[i].accum_counter, 1.0f);
        float grad = d_densification_info[i].accum_grad_mean2d / counter;
        float absgrad = d_densification_info[i].accum_absgrad_mean2d / counter;
        if (sanitize_non_finite) {
          if (!isfinite(grad)) grad = 0.0f;
          if (!isfinite(absgrad)) absgrad = 0.0f;
        }
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

  // Count clones and splits separately
  const int num_clones = thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] == kClone ? 1 : 0; },
      0, thrust::plus<int>());

  const int num_splits = thrust::transform_reduce(
      exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      [d_grow_flags] __device__(int i) -> int { return d_grow_flags[i] == kSplit ? 1 : 0; },
      0, thrust::plus<int>());

  if (m_print_verbose_stats) {
    const DuplicateVerboseStats verbose_stats = thrust::transform_reduce(
        exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        [d_densification_info, d_scale, d_importance,
         clone_thresh, split_thresh, scale_boundary, importance_thresh,
         sanitize_non_finite = m_sanitize_nan_gradients] __device__(int i) -> DuplicateVerboseStats {
          DuplicateVerboseStats out{};
          out.total = 1;
          const float raw_counter = d_densification_info[i].accum_counter;
          if (raw_counter <= 0.0f) out.counter_non_positive = 1;
          const float counter = fmaxf(raw_counter, 1.0f);

          float grad = d_densification_info[i].accum_grad_mean2d / counter;
          float absgrad = d_densification_info[i].accum_absgrad_mean2d / counter;
          if (!isfinite(grad)) out.grad_non_finite_before_sanitize = 1;
          if (!isfinite(absgrad)) out.absgrad_non_finite_before_sanitize = 1;
          if (sanitize_non_finite) {
            if (!isfinite(grad)) grad = 0.0f;
            if (!isfinite(absgrad)) absgrad = 0.0f;
          }

          const float max_scale = max(activate_scale(d_scale[i]));
          const bool is_small = max_scale <= scale_boundary;
          if (is_small) {
            out.small_scale = 1;
          } else {
            out.large_scale = 1;
          }

          if (grad >= clone_thresh) out.grad_pass = 1;
          if (absgrad >= split_thresh) out.absgrad_pass = 1;

          bool importance_ok = true;
          if (d_importance == nullptr) {
            out.importance_unavailable = 1;
          } else {
            importance_ok = d_importance[i] > importance_thresh;
            if (importance_ok) {
              out.importance_pass = 1;
            } else {
              out.importance_fail = 1;
            }
          }

          if (!importance_ok) {
            out.reject_importance = 1;
          } else if (is_small) {
            if (grad >= clone_thresh) {
              out.clone = 1;
            } else {
              out.reject_small_grad = 1;
            }
          } else {
            if (absgrad >= split_thresh) {
              out.split = 1;
            } else {
              out.reject_large_absgrad = 1;
            }
          }
          return out;
        },
        DuplicateVerboseStats{},
        DuplicateVerboseStatsAdd{});

    printf(
        "[FastGS][Verbose] duplicate thresholds: clone_grad>=%.6g split_absgrad>=%.6g "
        "scale_boundary=%.6g importance_thresh=%.6g has_importance=%d\n",
        clone_thresh,
        split_thresh,
        scale_boundary,
        importance_thresh,
        has_importance);
    printf(
        "[FastGS][Verbose] duplicate decisions: total=%lld clone=%lld split=%lld "
        "reject_importance=%lld reject_small_scale_grad=%lld reject_large_scale_absgrad=%lld\n",
        verbose_stats.total,
        verbose_stats.clone,
        verbose_stats.split,
        verbose_stats.reject_importance,
        verbose_stats.reject_small_grad,
        verbose_stats.reject_large_absgrad);
    printf(
        "[FastGS][Verbose] duplicate predicate coverage: "
        "counter<=0=%lld importance(pass/fail/unavailable)=%lld/%lld/%lld "
        "scale(small/large)=%lld/%lld grad_pass=%lld absgrad_pass=%lld "
        "non_finite_before_sanitize(grad/absgrad)=%lld/%lld\n",
        verbose_stats.counter_non_positive,
        verbose_stats.importance_pass,
        verbose_stats.importance_fail,
        verbose_stats.importance_unavailable,
        verbose_stats.small_scale,
        verbose_stats.large_scale,
        verbose_stats.grad_pass,
        verbose_stats.absgrad_pass,
        verbose_stats.grad_non_finite_before_sanitize,
        verbose_stats.absgrad_non_finite_before_sanitize);

    if (has_importance) {
      const float* d_importance_score = thrust::raw_pointer_cast(m_importance_score.data());
      const NumericMoments importance_moments = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_importance_score,
          static_cast<int>(num_gaussians));
      const long long selected_by_importance = count_above_threshold(
          to_cuda_stream(ctx.queue),
          d_importance_score,
          static_cast<int>(num_gaussians),
          importance_thresh);
      log_numeric_moments("duplicate.importance_score", importance_moments);
      printf(
          "[FastGS][Verbose] duplicate.importance_score >%.6g: %lld / %zu\n",
          importance_thresh,
          selected_by_importance,
          num_gaussians);
    }
    if (m_pruning_score.size() == num_gaussians) {
      const float* d_pruning_score = thrust::raw_pointer_cast(m_pruning_score.data());
      const NumericMoments pruning_moments = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_pruning_score,
          static_cast<int>(num_gaussians));
      log_numeric_moments("duplicate.pruning_score", pruning_moments);
    }
  }

  if (num_clones == 0 && num_splits == 0) return;

  // Reference: net change = +num_clones + 2*num_splits - num_splits = +num_clones + num_splits
  log_info("[FastGS] Densify: {} clone, {} split (N=2 new + remove orig), net +{}",
           num_clones, num_splits, num_clones + num_splits);

  // ===========================================================================
  // Phase 1: Clone — add 1 new Gaussian per clone source, copy all raw params
  // Reference: densify_and_clone_fastgs copies _xyz, _opacity, _scaling, _rotation, features
  // ===========================================================================
  if (num_clones > 0) {
    thrust::device_vector<int> clone_src(num_clones);
    thrust::copy_if(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        d_grow_flags, clone_src.data(),
        [] __device__(char f) { return f == 1; });

    thrust::device_vector<int> clone_target(num_clones);
    thrust::copy(exec,
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians) + num_clones),
        clone_target.data());

    // on_duplicate appends buffer space and copies optimizer state
    StrategyBase::on_duplicate(
        thrust::raw_pointer_cast(clone_src.data()),
        thrust::raw_pointer_cast(clone_target.data()),
        num_clones,
        ctx.queue);

    // Copy all raw parameters for clones (in deactivated space, matching reference)
    const int n_after_clone = static_cast<int>(m_gaussians->size());
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(num_clones),
        [d_clone_src = thrust::raw_pointer_cast(clone_src.data()),
         d_clone_tgt = thrust::raw_pointer_cast(clone_target.data()),
         means3d = thrust::raw_pointer_cast(m_gaussians->means().data()),
         scales3d = thrust::raw_pointer_cast(m_gaussians->scales().data()),
         opacities = thrust::raw_pointer_cast(m_gaussians->opacities().data()),
         rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
         sh0 = thrust::raw_pointer_cast(m_gaussians->sh0().data()),
         sh1 = thrust::raw_pointer_cast(m_gaussians->sh1().data()),
         sh2 = thrust::raw_pointer_cast(m_gaussians->sh2().data()),
         sh3 = thrust::raw_pointer_cast(m_gaussians->sh3().data()),
         n_after_clone] __device__(int i) {
          const int src = d_clone_src[i];
          const int tgt = d_clone_tgt[i];
          means3d[tgt] = means3d[src];
          scales3d[tgt] = scales3d[src];
          opacities[tgt] = opacities[src];
          rotations[tgt] = rotations[src];
          // SH data is SoA with stride = current buffer size
          for (int ch = 0; ch < 3; ch++)   sh0[ch * n_after_clone + tgt] = sh0[ch * n_after_clone + src];
          for (int ch = 0; ch < 9; ch++)   sh1[ch * n_after_clone + tgt] = sh1[ch * n_after_clone + src];
          for (int ch = 0; ch < 15; ch++)  sh2[ch * n_after_clone + tgt] = sh2[ch * n_after_clone + src];
          for (int ch = 0; ch < 21; ch++)  sh3[ch * n_after_clone + tgt] = sh3[ch * n_after_clone + src];
        });
  }

  // ===========================================================================
  // Phase 2: Split — create N=2 new Gaussians per split source with Gaussian
  //          random offsets, then REMOVE the originals.
  // Reference: densify_and_split_fastgs
  //   samples  = Normal(0, activated_scale)
  //   new_xyz  = rot @ samples + original_xyz        (for each of N=2 copies)
  //   new_scale = scaling_inverse_activation(activated_scale / (0.8*N))
  //   new_opacity = _opacity  (raw, unchanged)
  //   new_rotation = _rotation (raw, unchanged)
  //   Then prune originals.
  // ===========================================================================
  if (num_splits > 0) {
    const int size_before_split = static_cast<int>(m_gaussians->size());

    // Collect split source indices (still in original [0, num_gaussians) range)
    thrust::device_vector<int> split_src(num_splits);
    thrust::copy_if(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        d_grow_flags, split_src.data(),
        [] __device__(char f) { return f == 2; });

    // Expand source indices: [s0, s0, s1, s1, ...] — each source repeated N=2 times
    const int num_new_splits = 2 * num_splits;
    thrust::device_vector<int> split_src_expanded(num_new_splits);
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(num_new_splits),
        [d_src = thrust::raw_pointer_cast(split_src.data()),
         d_exp = thrust::raw_pointer_cast(split_src_expanded.data())] __device__(int i) {
          d_exp[i] = d_src[i / 2];
        });

    // Target indices: contiguous from size_before_split
    thrust::device_vector<int> split_target(num_new_splits);
    thrust::copy(exec,
        thrust::make_counting_iterator<int>(size_before_split),
        thrust::make_counting_iterator<int>(size_before_split + num_new_splits),
        split_target.data());

    // on_duplicate appends buffer space and copies optimizer state
    StrategyBase::on_duplicate(
        thrust::raw_pointer_cast(split_src_expanded.data()),
        thrust::raw_pointer_cast(split_target.data()),
        num_new_splits,
        ctx.queue);

    // Generate N(0,1) random samples for position offsets: 3 floats per new Gaussian
    // Reference: samples = torch.normal(mean=0, std=activated_scale) = N(0,1) * activated_scale
    thrust::device_vector<float> rng_buf(num_new_splits * 3);
    generate_random_normal(m_rng, num_new_splits * 3,
        thrust::raw_pointer_cast(rng_buf.data()), 0.0f, 1.0f);

    // Fill in parameters for the 2*num_splits new Gaussians
    const int n_after_split = static_cast<int>(m_gaussians->size());
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(num_new_splits),
        [d_split_src = thrust::raw_pointer_cast(split_src_expanded.data()),
         d_split_tgt = thrust::raw_pointer_cast(split_target.data()),
         means3d = thrust::raw_pointer_cast(m_gaussians->means().data()),
         scales3d = thrust::raw_pointer_cast(m_gaussians->scales().data()),
         opacities = thrust::raw_pointer_cast(m_gaussians->opacities().data()),
         rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
         sh0 = thrust::raw_pointer_cast(m_gaussians->sh0().data()),
         sh1 = thrust::raw_pointer_cast(m_gaussians->sh1().data()),
         sh2 = thrust::raw_pointer_cast(m_gaussians->sh2().data()),
         sh3 = thrust::raw_pointer_cast(m_gaussians->sh3().data()),
         n_after_split,
         rng = thrust::raw_pointer_cast(rng_buf.data())
        ] __device__(int i) {
          const int src = d_split_src[i];
          const int tgt = d_split_tgt[i];

          // Build rotation matrix from source quaternion (w, x, y, z) stored as (x,y,z,w) in vec4
          float r = rotations[src].x;  // w
          float x = rotations[src].y;  // x
          float y = rotations[src].z;  // y
          float z = rotations[src].w;  // z
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

          // Reference: samples = normal(0, activated_scale), offset = rot @ samples
          const vec3 actual_scale = activate_scale(scales3d[src]);
          const vec3 normal_sample = vec3(rng[i * 3 + 0], rng[i * 3 + 1], rng[i * 3 + 2]);
          const vec3 offset = rot * (normal_sample * actual_scale);

          // new_xyz = original_xyz + offset
          means3d[tgt] = means3d[src] + offset;
          // new_scaling = scaling_inverse_activation(activated_scale / (0.8 * N=2)) = deactivate(scale/1.6)
          scales3d[tgt] = deactivate_scale(actual_scale / 1.6f);
          // Reference copies raw _opacity unchanged (no sqrt trick)
          opacities[tgt] = opacities[src];
          // Copy raw rotation unchanged
          rotations[tgt] = rotations[src];
          // Copy SH per-degree in SoA layout with stride = n_after_split
          for (int ch = 0; ch < 3; ch++)   sh0[ch * n_after_split + tgt] = sh0[ch * n_after_split + src];
          for (int ch = 0; ch < 9; ch++)   sh1[ch * n_after_split + tgt] = sh1[ch * n_after_split + src];
          for (int ch = 0; ch < 15; ch++)  sh2[ch * n_after_split + tgt] = sh2[ch * n_after_split + src];
          for (int ch = 0; ch < 21; ch++)  sh3[ch * n_after_split + tgt] = sh3[ch * n_after_split + src];
        });

    // Remove the original split sources (reference: prune_points(selected_pts_mask))
    const int size_total = static_cast<int>(m_gaussians->size());
    thrust::device_vector<char> is_alive(size_total, 1);
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(num_splits),
        [d_alive = is_alive.data(),
         d_split_orig = thrust::raw_pointer_cast(split_src.data())] __device__(int i) {
          d_alive[d_split_orig[i]] = 0;
        });

    const int num_kept = size_total - num_splits;
    this->on_remove(thrust::raw_pointer_cast(is_alive.data()), num_kept, ctx.queue);
  }
}

// ---------------------------------------------------------------------------
// prune()  — budget-based pruning
// ---------------------------------------------------------------------------

void FastGSStrategy::prune(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
  const auto num_gaussians = m_gaussians->size();
  const int original_num_gaussians = buffer_count<DensificationInfo>(ctx.densification_info);
  const auto abs_ss_threshold = max(ctx.fwd_input.width, ctx.fwd_input.height) * m_params.max_screen_size;
  const bool prune_large = this_step() > m_params.reset_every;

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
        prune_degenerate_rotation = m_prune_degenerate_rotation,
pruning_scale_threshold = m_params.pruning_scale_threshold,
        prune_large,
        prune_large_ss = m_prune_large_ss,
        max_radii_threshold = abs_ss_threshold,
       original_num_gaussians,
       deninfo = buffer_data<DensificationInfo>(ctx.densification_info),
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) {
        bool not_large_ws = max(activate_scale(scale[i])) < pruning_scale_threshold * scene_scale;
        bool not_large_ss = !prune_large_ss || i >= original_num_gaussians ||
                            deninfo[i].max_radii_screen < max_radii_threshold;
        bool not_transparent = activate_opacity(d_opacity[i]) > min_opacity;
        bool not_degenerate = sum(abs(rotation[i])) > FLT_EPSILON;
        bool degen_ok = !prune_degenerate_rotation || not_degenerate;

        // Mark Gaussians that SHOULD be pruned = 1
        if (!(not_transparent && ((not_large_ws && not_large_ss) || !prune_large) && degen_ok)) {
          d_prune[i] = 1;
        }
      });

  // Count standard candidates
  int num_standard_candidates = thrust::reduce(exec,
      standard_prune.begin(), standard_prune.end(), 0, thrust::plus<int>());
  const bool has_pruning_scores = (m_pruning_score.size() == num_gaussians);
  char* d_standard_prune = thrust::raw_pointer_cast(standard_prune.data());

  if (m_print_verbose_stats) {
    printf(
        "[FastGS][Verbose] prune setup: total=%zu candidates=%d budget_ratio=%.6g prune_large=%d "
        "prune_large_ss=%d prune_degenerate_rotation=%d has_pruning_scores=%d\n",
        num_gaussians,
        num_standard_candidates,
        m_prune_budget_ratio,
        prune_large,
        m_prune_large_ss,
        m_prune_degenerate_rotation,
        has_pruning_scores);
    const PruneReasonStats candidate_reasons = thrust::transform_reduce(
        exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        [d_standard_prune,
         d_opacity,
         scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),
         scene_scale = m_gaussians->scene_scale(),
         rotation = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
         prune_degenerate_rotation = m_prune_degenerate_rotation,
         pruning_scale_threshold = m_params.pruning_scale_threshold,
         prune_large,
         prune_large_ss = m_prune_large_ss,
         max_radii_threshold = abs_ss_threshold,
         original_num_gaussians,
         deninfo = buffer_data<DensificationInfo>(ctx.densification_info),
         min_opacity = m_params.pruning_opacity_threshold] __device__(int i) -> PruneReasonStats {
          PruneReasonStats out{};
          if (d_standard_prune[i] == 0) return out;

          const bool low_opacity = activate_opacity(d_opacity[i]) <= min_opacity;
          const bool large_ws = prune_large &&
                                (max(activate_scale(scale[i])) >= pruning_scale_threshold * scene_scale);
          const bool large_ss = prune_large && prune_large_ss && i < original_num_gaussians &&
                                deninfo[i].max_radii_screen >= max_radii_threshold;
          const bool degenerate_rotation = prune_degenerate_rotation &&
                                           (sum(abs(rotation[i])) <= FLT_EPSILON);
          const int reason_count =
              static_cast<int>(low_opacity) +
              static_cast<int>(large_ws) +
              static_cast<int>(large_ss) +
              static_cast<int>(degenerate_rotation);

          out.selected = 1;
          if (low_opacity) out.low_opacity = 1;
          if (large_ws) out.large_world = 1;
          if (large_ss) out.large_screen = 1;
          if (degenerate_rotation) out.degenerate_rotation = 1;

          if (reason_count >= 2) {
            out.multi_reason = 1;
          } else if (reason_count == 1) {
            if (low_opacity) out.only_low_opacity = 1;
            if (large_ws) out.only_large_world = 1;
            if (large_ss) out.only_large_screen = 1;
            if (degenerate_rotation) out.only_degenerate_rotation = 1;
          }
          return out;
        },
        PruneReasonStats{},
        PruneReasonStatsAdd{});
    log_prune_reason_stats("prune.candidate_reasons", candidate_reasons);

    if (has_pruning_scores) {
      const float* d_pruning_score = thrust::raw_pointer_cast(m_pruning_score.data());
      const NumericMoments all_score_stats = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_pruning_score,
          static_cast<int>(num_gaussians));
      const NumericMoments candidate_score_stats = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_pruning_score,
          static_cast<int>(num_gaussians),
          d_standard_prune,
          true);
      log_numeric_moments("prune.pruning_score_all", all_score_stats);
      log_numeric_moments("prune.pruning_score_candidates", candidate_score_stats);
    }
  }

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
  const char* prune_mode = "fallback_all_candidates";
  int target_pruned = num_standard_candidates;

  if (has_pruning_scores && m_use_multinomial_pruning && budget < num_standard_candidates) {
    prune_mode = "multinomial_without_replacement";
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
        float score = d_ps[i];
        if (!isfinite(score)) score = 0.0f;
        score = fminf(fmaxf(score, 0.0f), 1.0f);
        return 1.0f / (1e-6f + 1.0f - score);
        });

    // Sample candidate positions by weight, without replacement.
    const int seed = static_cast<int>(m_rng.next_uint());
    auto sampled_positions_buf = multinomial_cuda_cpu_without_replacement(
      thrust::raw_pointer_cast(prune_weights.data()),
      num_standard_candidates,
      budget,
      seed,
      ctx.queue.get());

    // Mark sampled candidates as dead.
    const int actual_prune = static_cast<int>(buffer_count<int>(sampled_positions_buf));
    target_pruned = actual_prune;
    const int* d_sampled = buffer_data<int>(sampled_positions_buf);
    thrust::for_each(exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(actual_prune),
        [d_is_alive = is_alive.data(),
         d_candidate = thrust::raw_pointer_cast(candidate_indices.data()),
         d_sampled] __device__(int i) {
        const int candidate_idx = d_sampled[i];  // index into candidate_indices
          const int gaussian_idx = d_candidate[candidate_idx];
          d_is_alive[gaussian_idx] = 0;
        });
    } else if (has_pruning_scores && !m_use_multinomial_pruning && budget < num_standard_candidates) {
    prune_mode = "deterministic_top_weight";
    // Deterministic fallback: sort candidates by weight and take top budget.
    thrust::device_vector<int> candidate_indices(num_standard_candidates);
    thrust::copy_if(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
      standard_prune.begin(),
      candidate_indices.begin(),
      [] __device__(char p) { return p == 1; });

    thrust::device_vector<float> prune_weights(num_standard_candidates);
    const float* d_ps = thrust::raw_pointer_cast(m_pruning_score.data());
    thrust::transform(exec,
      candidate_indices.begin(), candidate_indices.end(),
      prune_weights.begin(),
      [d_ps] __device__(int i) -> float {
        float score = d_ps[i];
        if (!isfinite(score)) score = 0.0f;
        score = fminf(fmaxf(score, 0.0f), 1.0f);
        return 1.0f / (1e-6f + 1.0f - score);
      });

    thrust::device_vector<int> sorted_indices(num_standard_candidates);
    thrust::sequence(exec, sorted_indices.begin(), sorted_indices.end());
    thrust::sort_by_key(exec, prune_weights.begin(), prune_weights.end(),
      sorted_indices.begin(), thrust::greater<float>());

    const int actual_prune = std::min(budget, num_standard_candidates);
    target_pruned = actual_prune;
    thrust::for_each(exec,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(actual_prune),
      [d_is_alive = is_alive.data(),
       d_candidate = thrust::raw_pointer_cast(candidate_indices.data()),
       d_sorted = thrust::raw_pointer_cast(sorted_indices.data())] __device__(int i) {
        const int candidate_idx = d_sorted[i];
        const int gaussian_idx = d_candidate[candidate_idx];
        d_is_alive[gaussian_idx] = 0;
      });
  } else {
    log_warning("[FastGS] No pruning scores or budget >= candidates ({}), pruning all standard candidates.",
                num_standard_candidates);
    target_pruned = num_standard_candidates;
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
  if (m_print_verbose_stats) {
    const char* d_is_alive = thrust::raw_pointer_cast(is_alive.data());
    const PruneReasonStats removed_reasons = thrust::transform_reduce(
        exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        [d_is_alive,
         d_opacity,
         scale = thrust::raw_pointer_cast(m_gaussians->scales().data()),
         scene_scale = m_gaussians->scene_scale(),
         rotation = thrust::raw_pointer_cast(m_gaussians->rotations().data()),
         prune_degenerate_rotation = m_prune_degenerate_rotation,
         pruning_scale_threshold = m_params.pruning_scale_threshold,
         prune_large,
         prune_large_ss = m_prune_large_ss,
         max_radii_threshold = abs_ss_threshold,
         original_num_gaussians,
         deninfo = buffer_data<DensificationInfo>(ctx.densification_info),
         min_opacity = m_params.pruning_opacity_threshold] __device__(int i) -> PruneReasonStats {
          PruneReasonStats out{};
          if (d_is_alive[i] != 0) return out;

          const bool low_opacity = activate_opacity(d_opacity[i]) <= min_opacity;
          const bool large_ws = prune_large &&
                                (max(activate_scale(scale[i])) >= pruning_scale_threshold * scene_scale);
          const bool large_ss = prune_large && prune_large_ss && i < original_num_gaussians &&
                                deninfo[i].max_radii_screen >= max_radii_threshold;
          const bool degenerate_rotation = prune_degenerate_rotation &&
                                           (sum(abs(rotation[i])) <= FLT_EPSILON);
          const int reason_count =
              static_cast<int>(low_opacity) +
              static_cast<int>(large_ws) +
              static_cast<int>(large_ss) +
              static_cast<int>(degenerate_rotation);

          out.selected = 1;
          if (low_opacity) out.low_opacity = 1;
          if (large_ws) out.large_world = 1;
          if (large_ss) out.large_screen = 1;
          if (degenerate_rotation) out.degenerate_rotation = 1;

          if (reason_count >= 2) {
            out.multi_reason = 1;
          } else if (reason_count == 1) {
            if (low_opacity) out.only_low_opacity = 1;
            if (large_ws) out.only_large_world = 1;
            if (large_ss) out.only_large_screen = 1;
            if (degenerate_rotation) out.only_degenerate_rotation = 1;
          }
          return out;
        },
        PruneReasonStats{},
        PruneReasonStatsAdd{});
    printf(
        "[FastGS][Verbose] prune selection mode=%s budget=%d target_pruned=%d actually_removed=%zu\n",
        prune_mode,
        budget,
        target_pruned,
        num_gaussians - nums_kept);
    log_prune_reason_stats("prune.removed_reasons", removed_reasons);

    if (has_pruning_scores) {
      const float* d_pruning_score = thrust::raw_pointer_cast(m_pruning_score.data());
      const NumericMoments removed_score_stats = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_pruning_score,
          static_cast<int>(num_gaussians),
          d_is_alive,
          false);
      log_numeric_moments("prune.pruning_score_removed", removed_score_stats);
      const long long removed_score_gt_09 = count_above_threshold(
          to_cuda_stream(ctx.queue),
          d_pruning_score,
          static_cast<int>(num_gaussians),
          0.9f,
          d_is_alive,
          false);
      printf("[FastGS][Verbose] prune.removed pruning_score >0.9: %lld\n", removed_score_gt_09);
    }
  }
  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept, ctx.queue);
}

// ---------------------------------------------------------------------------
// final_prune() — periodic hard pruning based on pruning_score and opacity
// ---------------------------------------------------------------------------

void FastGSStrategy::final_prune(const RasterizeContext& ctx) {
  NVTX3_FUNC_RANGE();
  auto exec = thrust::cuda::par.on(to_cuda_stream(ctx.queue));
  const auto num_gaussians = m_gaussians->size();
  const bool has_pruning = (m_pruning_score.size() == num_gaussians);

  thrust::device_vector<char> is_alive(num_gaussians, 1);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  const float* d_ps = has_pruning ? thrust::raw_pointer_cast(m_pruning_score.data()) : nullptr;
  const float score_thresh = m_final_prune_score_threshold;
  const float opacity_thresh = m_final_prune_opacity_threshold;
  if (m_print_verbose_stats) {
    printf(
        "[FastGS][Verbose] final_prune setup: total=%zu has_pruning_scores=%d score_thresh=%.6g "
        "opacity_thresh=%.6g\n",
        num_gaussians,
        has_pruning,
        score_thresh,
        opacity_thresh);
    if (has_pruning) {
      const NumericMoments score_stats = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_ps,
          static_cast<int>(num_gaussians));
      log_numeric_moments("final_prune.pruning_score_all", score_stats);
      const long long score_gt_thresh = count_above_threshold(
          to_cuda_stream(ctx.queue),
          d_ps,
          static_cast<int>(num_gaussians),
          score_thresh);
      printf(
          "[FastGS][Verbose] final_prune.pruning_score >%.6g: %lld / %zu\n",
          score_thresh,
          score_gt_thresh,
          num_gaussians);
    }
  }

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
  if (m_print_verbose_stats) {
    const char* d_is_alive = thrust::raw_pointer_cast(is_alive.data());
    const FinalPruneReasonStats reason_stats = thrust::transform_reduce(
        exec,
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(static_cast<int>(num_gaussians)),
        [d_is_alive, d_opacity, d_ps, score_thresh, opacity_thresh] __device__(int i) -> FinalPruneReasonStats {
          FinalPruneReasonStats out{};
          if (d_is_alive[i] != 0) return out;
          const bool low_opacity = activate_opacity(d_opacity[i]) < opacity_thresh;
          const bool high_pruning_score = (d_ps != nullptr && d_ps[i] > score_thresh);
          out.removed = 1;
          if (low_opacity) out.low_opacity = 1;
          if (high_pruning_score) out.high_pruning_score = 1;
          if (low_opacity && high_pruning_score) {
            out.both_low_and_high = 1;
          } else if (low_opacity) {
            out.only_low_opacity = 1;
          } else if (high_pruning_score) {
            out.only_high_pruning_score = 1;
          }
          return out;
        },
        FinalPruneReasonStats{},
        FinalPruneReasonStatsAdd{});
    log_final_prune_reason_stats(reason_stats);

    if (has_pruning) {
      const NumericMoments removed_score_stats = reduce_numeric_moments(
          to_cuda_stream(ctx.queue),
          d_ps,
          static_cast<int>(num_gaussians),
          d_is_alive,
          false);
      log_numeric_moments("final_prune.pruning_score_removed", removed_score_stats);
    }
  }
  this->on_remove(thrust::raw_pointer_cast(is_alive.data()), nums_kept, ctx.queue);

  // Reset densification info
  ctx.densification_info = create_device_buffer_for<DensificationInfo>(ctx.runtime, nums_kept);
  fill_buffer_zero_async(ctx.runtime, ctx.queue, ctx.densification_info);
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
  if (config.contains("normalize_metric_l1"))    m_normalize_metric_l1 = config["normalize_metric_l1"].get<bool>();
  if (config.contains("metric_num_cameras"))     m_metric_num_cameras = config["metric_num_cameras"].get<int>();
  if (config.contains("sample_cameras_without_replacement"))
    m_sample_cameras_without_replacement = config["sample_cameras_without_replacement"].get<bool>();
  if (config.contains("photometric_l1_weight"))  m_photometric_l1_weight = config["photometric_l1_weight"].get<float>();
  if (config.contains("photometric_ssim_weight")) m_photometric_ssim_weight = config["photometric_ssim_weight"].get<float>();
  if (config.contains("sanitize_nan_gradients")) m_sanitize_nan_gradients = config["sanitize_nan_gradients"].get<bool>();
  if (config.contains("importance_threshold"))   m_importance_threshold = config["importance_threshold"].get<float>();
  if (config.contains("prune_budget_ratio"))     m_prune_budget_ratio = config["prune_budget_ratio"].get<float>();
  if (config.contains("use_multinomial_pruning")) m_use_multinomial_pruning = config["use_multinomial_pruning"].get<bool>();
  if (config.contains("prune_degenerate_rotation")) m_prune_degenerate_rotation = config["prune_degenerate_rotation"].get<bool>();
  if (config.contains("prune_large_ss"))         m_prune_large_ss = config["prune_large_ss"].get<bool>();
  if (config.contains("print_verbose_stats"))    m_print_verbose_stats = config["print_verbose_stats"].get<bool>();
  if (config.contains("final_prune_score_threshold"))
    m_final_prune_score_threshold = config["final_prune_score_threshold"].get<float>();
  if (config.contains("final_prune_opacity_threshold"))
    m_final_prune_opacity_threshold = config["final_prune_opacity_threshold"].get<float>();
  if (config.contains("final_prune_start"))      m_final_prune_start = config["final_prune_start"].get<int>();
  if (config.contains("final_prune_end"))        m_final_prune_end = config["final_prune_end"].get<int>();
  if (config.contains("final_prune_every"))      m_final_prune_every = config["final_prune_every"].get<int>();
  if (config.contains("opacity_reset_value"))    m_opacity_reset_value = config["opacity_reset_value"].get<float>();

  const float photometric_sum = m_photometric_l1_weight + m_photometric_ssim_weight;
  if (!(photometric_sum > 0.0f) || !std::isfinite(photometric_sum)) {
    log_warning("[FastGS] Invalid photometric weights (l1={}, ssim={}); resetting to 0.8/0.2.",
        m_photometric_l1_weight, m_photometric_ssim_weight);
    m_photometric_l1_weight = 0.8f;
    m_photometric_ssim_weight = 0.2f;
  } else {
    m_photometric_l1_weight /= photometric_sum;
    m_photometric_ssim_weight /= photometric_sum;
  }
}

json FastGSStrategy::get_params() const {
  json params = StrategyBase::get_params();
  params["type"] = "fastgs";
  params["absgrad_threshold"] = m_absgrad_threshold;
  params["percent_dense"] = m_percent_dense;
  params["loss_thresh"] = m_loss_thresh;
  params["normalize_metric_l1"] = m_normalize_metric_l1;
  params["metric_num_cameras"] = m_metric_num_cameras;
  params["sample_cameras_without_replacement"] = m_sample_cameras_without_replacement;
  params["photometric_l1_weight"] = m_photometric_l1_weight;
  params["photometric_ssim_weight"] = m_photometric_ssim_weight;
  params["sanitize_nan_gradients"] = m_sanitize_nan_gradients;
  params["importance_threshold"] = m_importance_threshold;
  params["prune_budget_ratio"] = m_prune_budget_ratio;
  params["use_multinomial_pruning"] = m_use_multinomial_pruning;
  params["prune_degenerate_rotation"] = m_prune_degenerate_rotation;
  params["prune_large_ss"] = m_prune_large_ss;
  params["print_verbose_stats"] = m_print_verbose_stats;
  params["final_prune_score_threshold"] = m_final_prune_score_threshold;
  params["final_prune_opacity_threshold"] = m_final_prune_opacity_threshold;
  params["final_prune_start"] = m_final_prune_start;
  params["final_prune_end"] = m_final_prune_end;
  params["final_prune_every"] = m_final_prune_every;
  params["opacity_reset_value"] = m_opacity_reset_value;
  return params;
}

#undef m_importance_score
#undef m_pruning_score

}  // namespace tinygs
