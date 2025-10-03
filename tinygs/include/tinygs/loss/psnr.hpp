#pragma once

#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/loss/loss.hpp"

namespace tinygs {

/// @brief Peak Signal-to-Noise Ratio metric
class PsnrMetric : public MetricBase {
public:
  PsnrMetric() = default;

  /// @brief Evaluate PSNR metric
  /// @param pred Predicted image
  /// @param target Target image
  /// @return PSNR value
  float evaluate(Image pred, Image target) override;

  /// @brief Get name of metric function
  /// @return Name of metric function
  std::string name() const override { return "psnr"; }

private:
  GPUBuffer<float> m_sqr_diff;
};

}