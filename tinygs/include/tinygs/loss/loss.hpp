#pragma once

#include "tinygs/core/image.hpp"
namespace tinygs {

struct LossContext {
  Image pred;
  Image target;
  Image loss;
  Image grad;
  cudaStream_t stream = nullptr;
};

/// @brief Base class for loss function (per-pixel only)
class LossBase {
public:
  LossBase() = default;
  virtual ~LossBase() = default;

  /// @brief Accumulate loss and gradient
  /// @param ctx Loss context
  /// @param scale Scaling factor for loss and gradient
  virtual void evaluate(LossContext ctx, float scale) = 0;

  /// @brief Get name of loss function
  /// @return Name of loss function
  virtual std::string name() const = 0;
};

class MetricBase {
public:
  virtual ~MetricBase() = default;

  /// @brief Evaluate metric (no gradient computation)
  /// @param pred Predicted image
  /// @param target Target image
  /// @return Metric value
  virtual float evaluate(Image pred, Image target) = 0;

  /// @brief Get name of metric function
  /// @return Name of metric function
  virtual std::string name() const = 0;
};

/// @brief Create loss object
/// @param loss_type Type of loss ("l1", "fused_ssim", etc.)
std::unique_ptr<LossBase> create_loss(const std::string& loss_type);

/// @brief Create metric object
/// @param metric_type Type of metric ("psnr", "ssim", etc.)
std::unique_ptr<MetricBase> create_metric(const std::string& metric_type);

} // namespace tinygs