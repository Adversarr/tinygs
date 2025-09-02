#pragma once
#include "tinygs/loss/loss.hpp"

namespace tinygs {

template <typename T>
class FusedSSIMLoss final : public LossBase<T> {
public:

  void evaluate(LossContext<T> ctx) override;

private:
  float m_c1 = 0.01f * 0.01f;
  float m_c2 = 0.03f * 0.03f;
};

}