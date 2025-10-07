#pragma once

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/optim/optim.hpp"
namespace tinygs {

struct StrategyParams {
  /// @brief Prune transparent gaussians
  float pruning_opacity_threshold = 0.005f;
  /// @brief Prune large gaussians in world space
  float pruning_scale_threshold = 0.1f;
  /// @brief Prune large gaussians in view space (2D)
  int max_screen_size = 20;

  /// @brief Grow if gradient is large (Default Strategy)
  float duplicate_grad_threshold = 0.0002f;
  /// @brief Split if large gaussian is found (Default Strategy)
  float duplicate_scale_threshold = 0.01f;

  bool reset_reset_optimizer = false;
  bool absgrad = false; /// @brief Whether to use absolute gradient for duplication
  int refine_every = 100;
  int start_refine = 500;
  int end_refine = 15'000;
  int max_num_gaussians = 10'000'000;
  int reset_every = 3'000;
  uint64_t seed = 42;

  /// @brief Default constructor with default values
  StrategyParams() = default;

  /// @brief Construct from JSON configuration
  explicit StrategyParams(const json& config);

  /// @brief Convert parameters to JSON
  json to_json() const;

  /// @brief Load parameters from JSON
  void from_json(const json& config);
};

class StrategyBase {
public:
  /// @brief Construct strategy with gaussians, gradients, and optimizer
  /// @param gaussians GPU gaussians data
  /// @param gaussians_grad GPU gaussians gradients
  /// @param optimizer Optimizer for updating gaussians
  explicit StrategyBase(std::shared_ptr<GPUGaussian3d> gaussians, 
                       std::shared_ptr<GPUGaussian3d> gaussians_grad,
                       std::shared_ptr<OptimizerBase> optimizer);

  virtual ~StrategyBase() = default;

  /// @brief Execute one step of the strategy
  /// @param ctx Rasterization context containing densification info
  void step(const RasterizeContext& ctx);

  /// @brief Reset the strategy state
  virtual void reset() = 0;

  /// @brief Implementation-specific strategy step logic
  /// @param ctx Rasterization context for strategy decisions
  virtual void step_impl(const RasterizeContext& ctx) = 0;

  /// @brief Set strategy parameters from JSON configuration
  /// @param config JSON configuration containing strategy parameters
  virtual void set_params(const json& config);

  /// @brief Get current strategy parameters as JSON
  virtual json get_params() const;

protected:
  /// @brief Handle removal of gaussians and update optimizer state
  /// @param kept_flag Array indicating which gaussians to keep
  /// @param num_kept Number of gaussians being kept
  void on_remove(char* kept_flag, int num_kept);
  
  /// @brief Handle duplication of gaussians and update optimizer state
  /// @param indices Original gaussian indices
  /// @param new_indices New gaussian indices after duplication
  /// @param num_duplications Number of gaussians being duplicated
  void on_duplicate(int* indices, int* new_indices, int num_duplications);
  
  /// @brief Handle reset of specific gaussians in optimizer
  /// @param indices Indices of gaussians to reset
  /// @param num_reset Number of gaussians to reset
  void on_reset(int* indices, int num_reset);
  
  /// @brief Handle opacity reset for all gaussians
  void on_reset_opacity();

  /// @brief Get current step count
  int this_step() const noexcept { return m_step_count; }

  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUGaussian3d> m_gaussians_grad;
  std::shared_ptr<OptimizerBase> m_optimizer;
  StrategyParams m_params;
  
private:
  int m_step_count = 0;
};

/// @brief Create a strategy object
/// @param strategy_type The type of strategy to create
/// @param gaussians The gaussians to optimize
/// @param gaussians_grad The gradient of gaussians
/// @param optimizer The optimizer to use
std::unique_ptr<StrategyBase> create_strategy(const std::string& strategy_type,
                                            std::shared_ptr<GPUGaussian3d> gaussians,
                                            std::shared_ptr<GPUGaussian3d> gaussians_grad,
                                            std::shared_ptr<OptimizerBase> optimizer);

}  // namespace tinygs