#pragma once

#include "tinygs/optim/lr_scheduler.hpp"
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

  /**
   * @brief Reset the strategy, excluding optimizer and gaussians
   */
  void reset() override;

  void step_impl(const RasterizeContext& ctx) override;

protected:
  float m_noise_lr_init = 1.0e5f;
  float m_noise_lr_decay = 1.0f - 2.0e-4f; // after 5'000 step, decay to about 1/e~=0.36
  float m_noise_lr = m_noise_lr_init;
  float m_grow_ratio = 1.05f;

public:
  // NOTE: MCMC use relocate instead of pruning.
  // void prune(const RasterizeContext& ctx);

  void add_noise(const RasterizeContext& ctx);
  void add_new_gs(const RasterizeContext& ctx);
  void relocate(const RasterizeContext& ctx);
};

}  // namespace tinygs