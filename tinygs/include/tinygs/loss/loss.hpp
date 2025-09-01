#pragma once

namespace tinygs::loss {

/**
 * @brief Base class for loss function. (only per-pixel is supported)
 * 
 * @tparam T 
 */
template <typename T>
class LossBase {
public:
  LossBase() = default;
  virtual ~LossBase() = default;

  /**
   * @brief Evaluate loss function.
   * 
   * @param input 
   * @param output 
   * @return T 
   */
  void evaluate(
  );

};

}