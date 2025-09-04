#pragma once

#include "strategy/strategy.hpp"
namespace tinygs {

class MCMCStrategy : public StrategyBase {
public:
  MCMCStrategy(std::shared_ptr<GPUGaussian3d> gaussians);
  virtual ~MCMCStrategy();

  virtual void step(const RasterizeContext& ctx) override;
  virtual void reset() override;

  void set_noise_lr(float noise_lr);

protected:
  float m_noise_lr = 5e5;

public:
  void pruning(const RasterizeContext& ctx);
  void add_noise(const RasterizeContext& ctx);
  void add_new_gs(const RasterizeContext& ctx);
};

}  // namespace tinygs