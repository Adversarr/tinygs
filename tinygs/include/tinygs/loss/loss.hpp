#pragma once

#include "tinygs/core/image.hpp"
namespace tinygs {

struct LossContext {
  float scale = 1.0f;
  Image pred;
  Image target;
  Image loss;
  Image grad;
  cudaStream_t stream = nullptr;
};

/**
 * @brief Base class for loss function. (only per-pixel is supported)
 * 
 */
class LossBase {
public:
  LossBase() = default;
  virtual ~LossBase() = default;

  /**
   * @brief Accumulate loss and gradient.
   * 
   * @param pred The prediction image.
   * @param target The target image.
   * @param loss The loss image.
   * @param grad The gradient image.
   */
  virtual void evaluate(LossContext ctx) = 0;
};

class MetricBase {
public:
  virtual ~MetricBase() = default;

  /**
   * @brief Evaluate the metric. (no gradient computation)
   * 
   * @param pred The prediction image.
   * @param target The target image.
   * @return float The metric value.
   */
  virtual float evaluate(Image pred, Image target) = 0;
};

/**
 * @brief Create a loss object
 *
 * @param loss_type The type of loss to create ("l1", "fused_ssim", etc.)
 * @return std::unique_ptr<LossBase> The created loss
 */
std::unique_ptr<LossBase> create_loss(const std::string& loss_type);

/**
 * @brief Create a metric object
 *
 * @param metric_type The type of metric to create ("psnr", "ssim", etc.)
 * @return std::unique_ptr<MetricBase> The created metric
 */
std::unique_ptr<MetricBase> create_metric(const std::string& metric_type);

} // namespace tinygs