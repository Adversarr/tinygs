#pragma once
#include "tinygs/loss/loss.hpp"

namespace tinygs {

/// @brief Fused SSIM loss implementation
class FusedSSIMLoss final : public LossBase {
public:
  FusedSSIMLoss();
  virtual ~FusedSSIMLoss();

  void evaluate(LossContext ctx, float scale) override;

  struct Impl;

private:
  float m_c1 = 0.01f * 0.01f;
  float m_c2 = 0.03f * 0.03f;
  std::unique_ptr<Impl> m_impl;
};

} // namespace tinygs