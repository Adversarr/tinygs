#pragma once

#include "tinygs/core/image.hpp"
namespace tinygs {

struct LossContext {
  float scale = 1.0f;
  Image<float> pred;
  Image<const float> target;
  Image<float> loss;
  Image<float> grad;
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

} // namespace tinygs