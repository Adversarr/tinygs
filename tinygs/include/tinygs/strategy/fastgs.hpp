#pragma once
#include "tinygs/random/pcg32.hpp"
#include "tinygs/strategy/strategy.hpp"

namespace tinygs {

/// @brief FastGS densification strategy.
///
/// Uses multi-view metric scoring to identify Gaussians that should be densified.
/// Renders a set of random cameras, computes per-pixel L1 loss, thresholds it, and
/// accumulates per-Gaussian importance / pruning scores.  Densification candidates
/// must pass both the gradient threshold AND the importance score threshold.
///
/// Key additions over DefaultStrategy:
///   1. `compute_gaussian_score()` – renders M random cameras and builds importance scores.
///   2. Clone/Split filtered by importance_score > importance_threshold.
///   3. Pruning uses budget-based sampling proportional to (1 - pruning_score).
///   4. Final prune: remove Gaussians with pruning_score > 0.9 or opacity < 0.1.
///
/// Reference: ref_impl/FastGS/scene/gaussian_model.py
class FastGSStrategy : public StrategyBase {
public:
  explicit FastGSStrategy(std::shared_ptr<GPUGaussian3d> gaussians,
                          std::shared_ptr<GPUGaussian3d> gaussians_grad,
                          std::shared_ptr<OptimizerBase> optimizer);
  ~FastGSStrategy() override;

  void reset() override;
  void step_impl(const RasterizeContext& ctx) override;

  void set_params(const json& config) override;
  json get_params() const override;

  /// @brief Store the rasterizer for multi-view metric rendering.
  void set_rasterizer(std::shared_ptr<RasterizerBase> rasterizer) override;

  /// @brief Store the dataloader for random camera sampling.
  void set_dataloader(std::shared_ptr<DataLoaderBase> dataloader) override;

  /// @brief Render M random cameras and accumulate per-Gaussian importance / pruning scores.
  void compute_gaussian_score(const RasterizeContext& ctx);

  /// @brief Duplicate (clone + split) Gaussians selected by gradient and importance.
  void duplicate(const RasterizeContext& ctx);

  /// @brief Prune dead Gaussians using budget-based sampling proportional to pruning score.
  void prune(const RasterizeContext& ctx);

  /// @brief Hard prune: remove Gaussians with opacity < threshold or pruning_score > threshold.
  void final_prune(const RasterizeContext& ctx);

  // -- External components (set by Orchestrator) --
  std::shared_ptr<RasterizerBase> m_rasterizer;
  std::shared_ptr<DataLoaderBase> m_dataloader;

  pcg32 m_rng;

  // -- Per-Gaussian accumulated scores (persistent across refine_every interval) --
  thrust::device_vector<float> m_importance_score;   ///< Accumulated importance (metric count)
  thrust::device_vector<float> m_pruning_score;      ///< Accumulated pruning score

  // -- Hyper-parameters --
  float m_absgrad_threshold = 0.0012f;     ///< Split abs gradient threshold
  float m_percent_dense = 0.001f;          ///< Clone/split scale boundary
  float m_loss_thresh = 0.1f;             ///< Per-pixel L1 threshold for metric map
  int m_metric_num_cameras = 10;           ///< Number of random cameras for scoring
  float m_importance_threshold = 5.0f;     ///< Minimum importance score for densification
  float m_prune_budget_ratio = 0.5f;       ///< Fraction of standard prune candidates to remove
  float m_final_prune_score_threshold = 0.9f; ///< Pruning score above which Gaussians are removed
  float m_final_prune_opacity_threshold = 0.1f; ///< Opacity below which Gaussians are removed in final prune
  int m_final_prune_start = 15000;         ///< First step for final pruning
  int m_final_prune_end = 30000;           ///< Last step for final pruning
  int m_final_prune_every = 3000;          ///< Interval for final pruning
  float m_opacity_reset_value = 0.8f;      ///< After densification, clip opacity to this max
};

}  // namespace tinygs
