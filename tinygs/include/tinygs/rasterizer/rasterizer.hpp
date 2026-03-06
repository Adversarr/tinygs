#pragma once
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/core/gaussian.hpp"
#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/platform/backend_types.hpp"
#include "tinygs/platform/runtime.hpp"

namespace tinygs {

/// @brief Context object that carries all inputs, outputs, and intermediate state for a
///        forward/backward rasterization pass.
///
/// Lifecycle: Owned by the Orchestrator.  Fields are populated before calling forward(),
/// and gradient fields before calling backward().
///
/// Thread-safety: Not thread-safe.  All CUDA work is serialized on `stream`.
struct RasterizeContext {
  /// @brief When true, the backward pass also computes gradients w.r.t. camera
  ///        intrinsics (K) and extrinsics (w2c) in `grad_input`.
  bool prepare_input_gradients = false;

  /// @brief When true, the forward pass skips storing intermediate tensors needed
  ///        for back-propagation, reducing memory usage during inference.
  bool inference = false;

  /// @brief Backend stream on which all forward/backward kernels are launched.
  BackendStream stream = nullptr;

  /// @brief Backend runtime for buffer allocation and memory operations.
  std::shared_ptr<BackendRuntime> runtime;

  /// @brief Backend queue used for buffer operations matching the stream.
  std::shared_ptr<BackendQueue> queue;

  /// @brief Global gradient scaler applied during backward pass to stabilize
  ///        mixed-precision training (typically 128 for FP16, 1 for FP32).
  float grad_scaler = 1.0f;

  GPUBatchInput fwd_input;    ///< Camera params + image dims for the current sample
  GPUBatchOutput fwd_output;  ///< Rendered image produced by forward()
  GPUBatchInput grad_input;   ///< Gradients w.r.t. camera params (populated by backward())
  GPUBatchOutput grad_output; ///< dL/d(rendered_image); must be set before backward()
  std::shared_ptr<GPUGaussian3d> gaussians_grad; ///< Accumulated Gaussian parameter gradients

  /// @brief Per-Gaussian densification statistics (view-space radii, accumulated
  ///        gradients, etc.) produced by forward() and consumed by Strategy.
  mutable std::shared_ptr<BackendBuffer> densification_info;

  // -- FastGS metric accumulation fields --

  /// @brief When true, the forward pass also accumulates per-Gaussian metric counts
  ///        for pixels flagged in `metric_map`. Used by FastGS strategy.
  bool metric_mode = false;

  /// @brief Per-pixel binary flag (H*W ints, FLAT row-major storage).
  ///        Pixels with value != 0 contribute to metric_counts for each Gaussian
  ///        that covers them.
  ///
  ///        IMPORTANT: Unlike other image buffers in this codebase which use 8x8
  ///        tiled storage, metric_map uses FLAT row-major indexing:
  ///          pixel_idx = y * width + x
  ///        This is intentional since metric_map is a flag array, not an image.
  mutable std::shared_ptr<BackendBuffer> metric_map;

  /// @brief Per-Gaussian metric counts (N ints). Incremented atomically during
  ///        metric_mode forward for each flagged pixel a Gaussian covers.
  mutable std::shared_ptr<BackendBuffer> metric_counts;
};

/// @brief Serializable parameters common to all rasterizer implementations.
struct RasterizerParams {
  DataType data_type = DataType::Float32; ///< Precision for rasterization buffers

  void from_json(const json& j);
  json to_json() const;
};

/// @brief Abstract base class for 3DGS rasterizer implementations.
///
/// Contract:
///   - `set_gaussians()` must be called once before the first `forward()`.
///   - `forward()` writes `ctx.fwd_output` and (unless `ctx.inference`) stores
///     intermediate tensors needed by `backward()`.
///   - `backward()` reads `ctx.grad_output` and fills `ctx.grad_input` +
///     `ctx.gaussians_grad`.  It must be called on the same context that was
///     last passed to `forward()`.
///   - All CUDA work is enqueued on `ctx.stream`.
class RasterizerBase {
public:
  explicit RasterizerBase(std::shared_ptr<BackendRuntime> runtime);
  
  /// @brief Constructor without runtime (for backward compatibility)
  RasterizerBase();

  virtual ~RasterizerBase() = default;

  /// @brief Get the backend runtime
  std::shared_ptr<BackendRuntime> runtime() const { return m_runtime; }

  /// @brief Render Gaussians into an image.
  /// @param params Fully-populated context (fwd_input must be set).
  virtual void forward(const RasterizeContext& params) = 0;

  /// @brief Compute parameter gradients given image-space loss gradients.
  /// @param params Context previously used in forward(); grad_output.image must be set.
  virtual void backward(RasterizeContext& params) = 0;

  /// @brief Forward pass with metric accumulation.  When ctx.metric_mode is true and
  ///        ctx.metric_map is provided, the rasterizer atomically increments
  ///        ctx.metric_counts for each Gaussian that covers a flagged pixel.
  ///        Default implementation simply delegates to forward().
  virtual void forward_metric(const RasterizeContext& params) { forward(params); }

  /// @brief Rebind the Gaussian data pointer (e.g. after densification resizes the buffer).
  virtual void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians);

  virtual json get_params() const = 0;
  virtual void set_params(const json& j) = 0;

protected:
  std::shared_ptr<BackendRuntime> m_runtime;
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  RasterizerParams m_params;
};

/// @brief Factory function for creating rasterizers.
/// @param rasterizer_type One of: "fastgs", "cpu".
/// @param runtime Backend runtime for GPU operations.
std::unique_ptr<RasterizerBase> create_rasterizer(const std::string& rasterizer_type,
                                                   std::shared_ptr<BackendRuntime> runtime);

}
