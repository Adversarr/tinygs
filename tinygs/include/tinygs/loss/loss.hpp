#pragma once

#include "core/image.hpp"
namespace tinygs {

template<typename T>
struct LossContext {
  float scale = 1.0f;
  Image<const T> pred;
  Image<const T> target;
  Image<T> loss;
  Image<T> grad;
  cudaStream_t stream = nullptr;
};

/**
 * @brief Base class for loss function. (only per-pixel is supported)
 * 
 * @tparam T The data type of the image.
 */
template <typename T>
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
  virtual void evaluate(LossContext<T> ctx) = 0;
};

} // namespace tinygs