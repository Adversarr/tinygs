#pragma once

#include <memory>

#include "tinygs/loss/loss.hpp"

namespace tinygs {

template <typename T>
class GPUBuffer;

/// @brief Peak Signal-to-Noise Ratio metric
class PsnrMetric : public MetricBase {
public:
  PsnrMetric();
  ~PsnrMetric() override;

  /// @brief Evaluate PSNR metric
  /// @param pred Predicted image
  /// @param target Target image
  /// @return PSNR value
  float evaluate(Image pred, Image target) override;

  /// @brief Get name of metric function
  /// @return Name of metric function
  std::string name() const override { return "psnr"; }

private:
  std::unique_ptr<GPUBuffer<float>> m_sqr_diff;
};

} 
