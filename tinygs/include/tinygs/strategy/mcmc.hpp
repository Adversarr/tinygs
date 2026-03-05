#pragma once
#include "tinygs/strategy/mcmc.hpp"
#include "tinygs/optim/lr_scheduler.hpp"
#include "tinygs/random/pcg32.hpp"
#include "tinygs/strategy/strategy.hpp"
namespace tinygs {

struct MCMCParams {
  /**
   * @note The standard noise_lr_init is 5e+5, and the final noise_lr is calculated:
   *          $$ noise_lr = noise_lr_init * mean3D_lr $$
   *       and mean3D_lr = 1.6e-4
   *       We use a different apporach: 
   *        - Learning Rate of Optimizer $gamma$: 1.0, with exponential decay
   *        - LR scaler $s$ for mean3D = 1.6e-4
   *        - noise_lr_init controls the noise added to mean3D
   *          $$ noise_lr = noise_lr_init * gamma $$
   *       To match the original design, we set
   *             noise_lr_init = 80 => 80 * 1 = 1.6e-4 * 1e+5
   * 
   */
  float noise_lr_init = 80.0f; // 1e5 * 1.6e-4
  float grow_ratio = 1.05f;

  /// @brief Default constructor with default values
  MCMCParams() = default;

  /// @brief Construct from JSON configuration
  explicit MCMCParams(const json& config);

  /// @brief Convert parameters to JSON
  json to_json() const;

  /// @brief Load parameters from JSON
  void from_json(const json& config);
};

class MCMCStrategy : public StrategyBase {
public:
  /// @brief Construct MCMC strategy with runtime, gaussians, gradients, and optimizer
  /// @param runtime Backend runtime for GPU operations
  /// @param gaussians GPU gaussians data
  /// @param gaussians_grad GPU gaussians gradients
  /// @param optimizer Optimizer for updating gaussians
  MCMCStrategy(std::shared_ptr<BackendRuntime> runtime,
               std::shared_ptr<GPUGaussian3d> gaussians,
               std::shared_ptr<GPUGaussian3d> gaussians_grad,
               std::shared_ptr<OptimizerBase> optimizer);
  virtual ~MCMCStrategy();

  /// @brief Reset the strategy, excluding optimizer and gaussians
  void reset() override;

  void step_impl(const RasterizeContext& ctx) override;

  /// @brief Set strategy parameters from JSON configuration
  /// @param config JSON configuration containing strategy parameters
  void set_params(const json& config) override;

  /// @brief Get current strategy parameters as JSON
  json get_params() const override;

protected:
  MCMCParams m_mcmc_params;  ///< MCMC parameters
  pcg32 m_rng;               ///< Random number generator

  struct Impl;
  std::unique_ptr<Impl> m_impl;

public:
  // NOTE: MCMC use relocate instead of pruning.
  // void prune(const RasterizeContext& ctx);

  void add_noise(const RasterizeContext& ctx);
  void add_new_gs(const RasterizeContext& ctx);
  void relocate(const RasterizeContext& ctx);
};

}  // namespace tinygs