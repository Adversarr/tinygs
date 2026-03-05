#pragma once
#include "loss.hpp"

namespace tinygs {

/// @brief L1 loss implementation
class L1Loss : public LossBase {
public:
  using LossBase::LossBase;
  ~L1Loss() = default;

  /// @brief Evaluate L1 loss and gradient
  /// @param ctx Loss context
  /// @param scale Scaling factor for loss and gradient
  void evaluate(LossContext ctx, float scale) override;

  /// @brief Get name of loss function
  /// @return Name of loss function
  std::string name() const override { return "l1"; }
};

} // namespace tinygs
