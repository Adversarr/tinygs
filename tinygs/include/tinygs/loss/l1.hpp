#pragma once
#include "loss.hpp"

namespace tinygs {

class L1Loss : public LossBase {
public:
  L1Loss() = default;
  ~L1Loss() = default;

  void evaluate(LossContext ctx) override;
};

} // namespace tinygs
