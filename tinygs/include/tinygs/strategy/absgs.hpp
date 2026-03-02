#pragma once
#include "tinygs/random/pcg32.hpp"
#include "tinygs/strategy/strategy.hpp"

namespace tinygs {

/// @brief AbsGS densification strategy.
///
/// Uses standard gradients for cloning (small Gaussians) and absolute gradients for
/// splitting (large Gaussians).  Algorithm from "AbsGS: Recovering Fine Details for
/// 3D Gaussian Splatting" — the key difference from DefaultStrategy is that the split
/// condition uses `accum_absgrad_mean2d` with a separate (typically higher) threshold,
/// while the clone condition uses `accum_grad_mean2d`.
///
/// Reference: ref_impl/AbsGS/scene/gaussian_model.py
class AbsGSStrategy : public StrategyBase {
public:
  explicit AbsGSStrategy(std::shared_ptr<GPUGaussian3d> gaussians,
                         std::shared_ptr<GPUGaussian3d> gaussians_grad,
                         std::shared_ptr<OptimizerBase> optimizer);
  ~AbsGSStrategy() override;

  void reset() override;
  void step_impl(const RasterizeContext& ctx) override;

  void set_params(const json& config) override;
  json get_params() const override;

  void duplicate(const RasterizeContext& ctx);
  void prune(const RasterizeContext& ctx);

  pcg32 m_rng;

  /// @brief Absolute gradient threshold for split (default 0.0004, from AbsGS paper).
  float m_absgrad_threshold = 0.0004f;

  /// @brief Percent-dense parameter controlling the clone/split boundary.
  ///        Gaussians with max_scale <= percent_dense * scene_scale are cloned;
  ///        those above are split.  Default 0.001 (from AbsGS reference).
  float m_percent_dense = 0.001f;
};

}  // namespace tinygs
