#pragma once
#include "tinygs/strategy/strategy.hpp"

namespace tinygs {
class DefaultStrategy : public StrategyBase {
public:
  /**
   * @brief Construct a new DefaultStrategy object with gaussians, gradients, and optimizer
   * @param gaussians Shared pointer to GPU gaussians data
   * @param gaussians_grad Shared pointer to GPU gaussians gradients
   * @param optimizer Shared pointer to optimizer for updating gaussians
   */
  explicit DefaultStrategy(std::shared_ptr<GPUGaussian3d> gaussians,
                          std::shared_ptr<GPUGaussian3d> gaussians_grad,
                          std::shared_ptr<OptimizerBase> optimizer);
  ~DefaultStrategy() override;
  void reset() override;

  void step_impl(const RasterizeContext& ctx) override;
  void duplicate(const RasterizeContext& ctx);
  void prune(const RasterizeContext& ctx);
};
}  // namespace tinygs