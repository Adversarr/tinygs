#pragma once
#include "loss.hpp"

namespace tinygs {

class HuberLoss : public LossBase {
public:
  using LossBase::LossBase;
  ~HuberLoss() = default;

  void evaluate(LossContext ctx, float scale) override;

  std::string name() const override { return "huber"; }

private:
  float m_delta = 0.01f; // default threshold
};

} // namespace tinygs