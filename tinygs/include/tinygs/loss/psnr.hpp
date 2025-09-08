#pragma once

#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/loss/loss.hpp"

namespace tinygs {

/**
 * @brief Peak Signal-to-Noise Ratio
 */
class PsnrMetric : public MetricBase {
public:
  PsnrMetric() = default;

  float evaluate(Image pred, Image target) override;

private:
  GPUBuffer<float> m_sqr_diff;
};

}