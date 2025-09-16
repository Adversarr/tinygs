#pragma once

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct GaussianOptimizationParams {
  // Shared parameters
  float max_grad_1 = 1.0f;
  bool skip_zero_grad = false;

  // Learning rates for different parameters
  float means_lr = 1.6e-4f;
  float shs_lr = 2.5e-3f;
  float opacities_lr = 5.0e-2f;
  float scales_lr = 5.0e-3f;
  float rotations_lr = 1.0e-3f;

  // L1 regularization
  float opacities_l1 = 0.0f; //0.01f;
  float scales_l1 = 0.0f;    //0.01f;

  /** @brief Default constructor with default values */
  GaussianOptimizationParams() = default;

  /** @brief Construct from JSON configuration */
  explicit GaussianOptimizationParams(const json& config);

  /** @brief Convert parameters to JSON */
  json to_json() const;

  /** @brief Load parameters from JSON */
  void from_json(const json& config);
};

class OptimizerBase {
public:
  OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  virtual ~OptimizerBase() = default;

  void set_lr(float new_lr);

  float get_lr() const;

  /**
   * @brief Set optimizer parameters from JSON configuration
   * 
   * @param config JSON configuration containing optimizer parameters
   */
  virtual void set_params(const json& config);

  /**
   * @brief Get current optimizer parameters as JSON
   * 
   * @return JSON object containing current optimizer parameters
   */
  virtual json get_params() const;

  /**
   * @brief Perform one optimization step.
   * 
   * @param scale The gradient scale.
   */
  virtual void step(float scale) = 0;

  virtual void reset();

  /**
   * @brief Pre-remove flagged gaussians, remove the gaussian's momentum buffer also.
   *
   * @param kept_flag
   * @param num_kept The number of gaussians to remove.
   */
  virtual void remove(char* kept_flag, int num_kept){}

  /**
   * @brief After the gaussian is duplicated, update the momentum buffers for them.
   *
   * @param indices The indices of gaussians before duplication.
   * @param new_indices The indices of gaussians after duplication.
   * @param num_duplicate The number of duplicated gaussians.
   */
  virtual void duplicate(int* indices, int* new_indices, int num_duplicate){}

  /**
   * @brief Reset the momentum buffers for gaussians.
   *
   * @param indices The indices of gaussians to reset.
   * @param num_reset The number of gaussians to reset.
   */
  virtual void reset(int* indices, int num_reset) = 0;


  // Fxxk the default strategy.
  virtual void reset_opacity() = 0;

protected:
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUGaussian3d> m_gaussians_grad;

  float m_global_lr = 1.0f;
  GaussianOptimizationParams m_params;
};

/**
 * @brief Create a optimizer object
 * 
 * @param optimizer_type The type of optimizer to create.
 * @param gaussians The gaussians to optimize.
 * @param gaussians_grad The gradient of gaussians.
 * @return std::unique_ptr<OptimizerBase> The created optimizer.
 */
std::unique_ptr<OptimizerBase> create_optimizer(const std::string& optimizer_type,
                                                std::shared_ptr<GPUGaussian3d> gaussians,
                                                std::shared_ptr<GPUGaussian3d> gaussians_grad);

}  // namespace tinygs