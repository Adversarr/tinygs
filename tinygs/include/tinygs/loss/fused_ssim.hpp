#pragma once
#include "tinygs/loss/loss.hpp"

namespace tinygs {

/// @brief Fused SSIM loss implementation
class FusedSSIMLoss final : public LossBase {
public:
  FusedSSIMLoss();
  virtual ~FusedSSIMLoss();

  /// @brief Evaluate Fused SSIM loss and gradient
  /// @param ctx Loss context
  /// @param scale Scaling factor for loss and gradient
  void evaluate(LossContext ctx, float scale) override;

  /// @brief Get name of loss function
  /// @return Name of loss function
  std::string name() const override { return "fused_ssim"; }
  struct Impl;

private:
  float m_c1 = 0.01f * 0.01f;
  float m_c2 = 0.03f * 0.03f;
  std::unique_ptr<Impl> m_impl;
};

} // namespace tinygs