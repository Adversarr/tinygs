#pragma once

#include "tinygs/core/image.hpp"
#include "tinygs/platform/backend_types.hpp"
namespace tinygs {

/// @brief Shared context for loss computation.
///
/// All images must have identical shape and reside in GPU memory.  The `loss`
/// buffer is *accumulated* across multiple LossBase::evaluate() calls within
/// the same training step, so the caller should zero it once at the start of
/// each step.  `grad` is similarly accumulated (dL/d(pred)).
struct LossContext {
  Image pred;                    ///< Predicted (rendered) image  [read-only]
  Image target;                  ///< Ground-truth image          [read-only]
  Image loss;                    ///< Per-pixel loss buffer       [read-write, accumulated]
  Image grad;                    ///< dL/d(pred) gradient buffer  [read-write, accumulated]
  BackendStream stream = nullptr; ///< Backend stream for all kernels
};

/// @brief Abstract base class for per-pixel loss functions.
///
/// Implementations must atomically accumulate into `ctx.loss` and `ctx.grad`
/// so that multiple loss components can be composed.
class LossBase {
public:
  LossBase() = default;
  virtual ~LossBase() = default;

  /// @brief Compute loss and accumulate scaled gradients into `ctx.grad`.
  /// @param ctx Loss context (pred, target, loss buffer, grad buffer).
  /// @param scale Scalar multiplier applied to both loss and gradient
  ///              (includes loss weight × grad_scaler).
  virtual void evaluate(LossContext ctx, float scale) = 0;

  /// @brief Human-readable name used for logging / CSV export.
  virtual std::string name() const = 0;
};

/// @brief Abstract base class for evaluation metrics (no gradient).
class MetricBase {
public:
  virtual ~MetricBase() = default;

  /// @brief Compute a scalar quality metric between predicted and target images.
  /// @param pred Predicted image (GPU memory).
  /// @param target Ground-truth image (GPU memory).
  /// @return Scalar metric value (higher is better for PSNR/SSIM).
  virtual float evaluate(Image pred, Image target) = 0;

  /// @brief Human-readable name used for logging / CSV export.
  virtual std::string name() const = 0;
};

/// @brief Create loss object.
/// @param loss_type One of: "l1", "fused_ssim".
std::unique_ptr<LossBase> create_loss(const std::string& loss_type);

/// @brief Create metric object.
/// @param metric_type One of: "psnr".
std::unique_ptr<MetricBase> create_metric(const std::string& metric_type);

} // namespace tinygs
