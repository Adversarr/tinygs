#pragma once

#include "tinygs/optim/lr_scheduler.hpp"
#include "tinygs/strategy/strategy.hpp"
namespace tinygs {

struct MCMCParams {
  float noise_lr_init = 1.6f; // 1e5 * 1.6e-4
  float noise_lr_decay = 1.0f - 2.0e-4f; // after 5'000 step, decay to about 1/e~=0.36
  float grow_ratio = 1.05f;

  /** @brief Default constructor with default values */
  MCMCParams() = default;

  /** @brief Construct from JSON configuration */
  explicit MCMCParams(const json& config);

  /** @brief Convert parameters to JSON */
  json to_json() const;

  /** @brief Load parameters from JSON */
  void from_json(const json& config);
};

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

  /**
   * @brief Set strategy parameters from JSON configuration
   *
   * @param config JSON configuration containing strategy parameters
   */
  void set_params(const json& config) override;

  /**
   * @brief Get current strategy parameters as JSON
   *
   * @return JSON object containing current strategy parameters
   */
  json get_params() const override;

protected:
  MCMCParams m_mcmc_params;
  float m_noise_lr;

public:
  // NOTE: MCMC use relocate instead of pruning.
  // void prune(const RasterizeContext& ctx);

  void add_noise(const RasterizeContext& ctx);
  void add_new_gs(const RasterizeContext& ctx);
  void relocate(const RasterizeContext& ctx);

};

}  // namespace tinygs