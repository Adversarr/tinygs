#pragma once
#include "tinygs/strategy/strategy.hpp"

namespace tinygs {
class DefaultStrategy : public StrategyBase {
public:
  explicit DefaultStrategy(std::shared_ptr<GPUGaussian3d> gaussians);
  ~DefaultStrategy() override;
  void reset() override;

  void step_impl(const RasterizeContext& ctx) override;
  thrust::device_vector<bool> duplicate(const RasterizeContext& ctx);
  void prune(const RasterizeContext& ctx, const thrust::device_vector<bool> & disable_prune);
};
}  // namespace tinygs