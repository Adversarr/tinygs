#pragma once
#include "tinygs/strategy/strategy.hpp"

namespace tinygs {
class DefaultStrategy : public StrategyBase {
public:
  explicit DefaultStrategy(std::shared_ptr<GPUGaussian3d> gaussians);
  virtual ~DefaultStrategy();

  void step(const RasterizeContext& ctx) override;
  void reset() override;

  virtual void duplicate(const RasterizeContext& ctx);
  virtual void prune(const RasterizeContext& ctx);
};
}  // namespace tinygs