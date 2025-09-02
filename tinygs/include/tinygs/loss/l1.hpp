#pragma once
#include "loss.hpp"

namespace tinygs {
template <typename T>
class L1Loss : public LossBase<T> {
public:
  L1Loss() = default;
  ~L1Loss() = default;

  void evaluate(LossContext<T> ctx) override;
};

} // namespace tinygs
