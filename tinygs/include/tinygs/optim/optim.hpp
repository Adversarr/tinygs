#pragma once

#include "tinygs/core/gpu_gaussian.hpp"

namespace tinygs {

struct GaussianOptimizationParams {
  // Shared parameters
  float max_grad_1 = 1.0f;
  bool skip_zero_grad = true;

  // Learning rates for different parameters
  float means_lr = 1.6e-4f;
  float shs_lr = 2.5e-3f;
  float opacities_lr = 5.0e-2f;
  float scales_lr = 5.0e-3f;
  float rotations_lr = 1.0e-3f;

  // L1 regularization
  float opacities_l1 = 0.00f; //0.01f;
  float scales_l1 = 0.00f;    //0.01f;
};

class OptimizerBase {
public:
  OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  virtual ~OptimizerBase() = default;

  virtual void step(float scale) = 0;

  virtual void reset(){}

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

protected:
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUGaussian3d> m_gaussians_grad;

  GaussianOptimizationParams m_params;
};

}  // namespace tinygs