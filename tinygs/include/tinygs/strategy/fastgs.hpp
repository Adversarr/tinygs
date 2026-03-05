#pragma once
#include "tinygs/random/pcg32.hpp"
#include "tinygs/strategy/strategy.hpp"
#include <memory>

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
  explicit FastGSStrategy(std::shared_ptr<BackendRuntime> runtime,
                           std::shared_ptr<GPUGaussian3d> gaussians,
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
  /// @param ctx   The training context (used for stream, grad_scaler, etc.)
  /// @param densify  If true, also computes importance_score (used for densification filtering).
  ///                 If false, only computes pruning_score (used for final_prune).
  void compute_gaussian_score(const RasterizeContext& ctx, bool densify);

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

  // -- Hyper-parameters --
  float m_absgrad_threshold = 0.0012f;     ///< Split abs gradient threshold
  float m_percent_dense = 0.001f;          ///< Clone/split scale boundary
  float m_loss_thresh = 0.1f;             ///< Per-pixel L1 threshold for metric map
  bool m_normalize_metric_l1 = true;       ///< Min-max normalize per-pixel L1 before thresholding
  int m_metric_num_cameras = 10;           ///< Number of random cameras for scoring
  bool m_sample_cameras_without_replacement = true; ///< Sample metric cameras without replacement
  float m_photometric_l1_weight = 0.8f;    ///< Photometric L1 mixture weight
  float m_photometric_ssim_weight = 0.2f;  ///< Photometric SSIM-loss mixture weight (1 - SSIM)
  bool m_sanitize_nan_gradients = true;    ///< Replace non-finite grad stats with zero
  float m_importance_threshold = 5.0f;     ///< Minimum importance score for densification
  float m_prune_budget_ratio = 0.5f;       ///< Fraction of standard prune candidates to remove
  bool m_use_multinomial_pruning = true;   ///< Use stochastic multinomial pruning (without replacement)
  bool m_prune_degenerate_rotation = false; ///< Prune degenerate rotation quaternions
  bool m_prune_large_ss = false;            ///< Prune large gaussians in screen-space (default: disabled)
  bool m_print_verbose_stats = false;       ///< Print detailed FastGS score/decision/prune statistics
  float m_final_prune_score_threshold = 0.9f; ///< Pruning score above which Gaussians are removed
  float m_final_prune_opacity_threshold = 0.1f; ///< Opacity below which Gaussians are removed in final prune
  int m_final_prune_start = 18000;         ///< First final-prune step (>15000 and divisible by 3000)
  int m_final_prune_end = 27000;           ///< Last final-prune step (<30000 and divisible by 3000)
  int m_final_prune_every = 3000;          ///< Interval for final pruning
  float m_opacity_reset_value = 0.01f;     ///< Periodic opacity reset clamp (ref: reset_opacity → min(opacity, 0.01))

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

}  // namespace tinygs
