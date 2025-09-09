#pragma once

#include "tinygs/strategy/strategy.hpp"
namespace tinygs {

class MCMCStrategy : public StrategyBase {
public:
  /**
   * @brief Construct a new MCMCStrategy object with gaussians, gradients, and optimizer
   * @param gaussians Shared pointer to GPU gaussians data
   * @param gaussians_grad Shared pointer to GPU gaussians gradients
   * @param optimizer Shared pointer to optimizer for updating gaussians
   */
  MCMCStrategy(std::shared_ptr<GPUGaussian3d> gaussians,
               std::shared_ptr<GPUGaussian3d> gaussians_grad,
               std::shared_ptr<OptimizerBase> optimizer);
  virtual ~MCMCStrategy();

  void reset() override;

  void set_noise_lr(float noise_lr);

  void step_impl(const RasterizeContext& ctx) override;
protected:
  float m_noise_lr = 1;

public:
  // NOTE: MCMC use relocate instead of pruning.
  // void prune(const RasterizeContext& ctx);

  void add_noise(const RasterizeContext& ctx);
  void add_new_gs(const RasterizeContext& ctx);
  void relocate(const RasterizeContext& ctx);
};

}  // namespace tinygs