#pragma once
#include "loss.hpp"

namespace tinygs {

/// @brief L1 loss implementation
class L1Loss : public LossBase {
public:
  L1Loss() = default;
  ~L1Loss() = default;

  void evaluate(LossContext ctx, float scale) override;
};

} // namespace tinygs
