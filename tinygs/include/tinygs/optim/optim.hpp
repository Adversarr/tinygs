#pragma once

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct GaussianOptimizationParams {
  /// Shared parameters
  float max_grad_1 = 1.0f;
  bool skip_zero_grad = false;

  /// Learning rates for different parameters
  float means_lr = 1.6e-4f;
  float shs_lr = 2.5e-3f;
  float opacities_lr = 5.0e-2f;
  float scales_lr = 5.0e-3f;
  float rotations_lr = 1.0e-3f;

  /// L1 regularization
  float opacities_l1 = 0.0f; //0.01f;
  float scales_l1 = 0.0f;    //0.01f;

  /// @brief Default constructor with default values
  GaussianOptimizationParams() = default;

  /// @brief Construct from JSON configuration
  explicit GaussianOptimizationParams(const json& config);

  /// @brief Convert parameters to JSON
  json to_json() const;

  /// @brief Load parameters from JSON
  void from_json(const json& config);
};

class OptimizerBase {
public:
  OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  virtual ~OptimizerBase() = default;

  void set_lr(float new_lr);

  float get_lr() const;

  /// @brief Set optimizer parameters from JSON
  virtual void set_params(const json& config);

  /// @brief Get current optimizer parameters as JSON
  virtual json get_params() const;

  /// @brief Perform one optimization step
  virtual void step(float scale) = 0;

  /// @brief Reset optimizer state
  virtual void reset();

  /// @brief Pre-remove flagged gaussians and their momentum buffers
  virtual void remove(char* kept_flag, int num_kept){}

  /// @brief Update momentum buffers after gaussian duplication
  virtual void duplicate(int* indices, int* new_indices, int num_duplicate){}

  /// @brief Reset momentum buffers for specified gaussians
  virtual void reset(int* indices, int num_reset) = 0;

  /// @brief Reset opacity implementation
  virtual void reset_opacity() = 0;

  /// @brief set the gaussians, and reset
  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  std::shared_ptr<GPUGaussian3d> get_gaussians() const { return m_gaussians; }
  std::shared_ptr<GPUGaussian3d> get_gaussians_grad() const { return m_gaussians_grad; }

protected:
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUGaussian3d> m_gaussians_grad;

  float m_global_lr = 1.0f;
  GaussianOptimizationParams m_params;
};

/// @brief Create optimizer object
/// @param optimizer_type Type of optimizer to create
/// @param gaussians Gaussians to optimize
/// @param gaussians_grad Gradient of gaussians
std::unique_ptr<OptimizerBase> create_optimizer(const std::string& optimizer_type,
                                                std::shared_ptr<GPUGaussian3d> gaussians,
                                                std::shared_ptr<GPUGaussian3d> gaussians_grad);

}  // namespace tinygs