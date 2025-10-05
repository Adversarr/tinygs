#pragma once
#include "tinygs/random/pcg32.hpp"
#include "tinygs/strategy/strategy.hpp"

namespace tinygs {
class ImprovedStrategy : public StrategyBase {
public:
  /// @brief Construct improved strategy with gaussians, gradients, and optimizer
  explicit ImprovedStrategy(std::shared_ptr<GPUGaussian3d> gaussians,
                          std::shared_ptr<GPUGaussian3d> gaussians_grad,
                          std::shared_ptr<OptimizerBase> optimizer);
  ~ImprovedStrategy() override;
  void reset() override;

  void step_impl(const RasterizeContext& ctx) override;
  void duplicate(const RasterizeContext& ctx, int budget);
  void prune(const RasterizeContext& ctx);

  /// @brief Set strategy parameters from JSON configuration
  void set_params(const json& config) override;

  /// @brief Get current strategy parameters as JSON
  json get_params() const override;

  pcg32 m_rng;

private:
  // Improved-specific parameters
  float m_split_distance = 0.3f;      // similar to Python split_distance
  float m_opacity_reduction = 0.75f;  // similar to Python opacity_reduction
};
}  // namespace tinygs