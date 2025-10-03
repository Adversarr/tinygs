#pragma once
#include "loss.hpp"

namespace tinygs {

class L2Loss : public LossBase {
public:
  L2Loss() = default;
  ~L2Loss() = default;

  void evaluate(LossContext ctx, float scale) override;

  std::string name() const override { return "l2"; }
};

} // namespace tinygs