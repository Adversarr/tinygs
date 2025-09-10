#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"

namespace tinygs {

struct SGDParameters {
  // No momentum parameters needed for basic SGD
};

/**
 * @brief Basic SGD optimizer without momentum for Gaussian Splatting
 * 
 * This optimizer applies gradient descent directly without maintaining any internal state.
 * It supports gradient scaling separate from learning rate scaling.
 */
class SGD : public OptimizerBase {
public:
  /**
   * @brief Construct SGD optimizer with specified parameters
   * @param gaussians Shared pointer to GPU Gaussian parameters
   * @param gaussians_grad Shared pointer to GPU Gaussian gradients
   * @param params SGD-specific parameters (currently none)
   */
  SGD(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad,
        const SGDParameters& params = {});

  ~SGD() override = default;

  /** @brief Reset optimizer state (no-op for SGD) */
  void reset() override;
  
  /** @brief Perform one optimization step with gradient scaling */
  void step(float scale) override;
  
  /** @brief Remove optimizer state (no-op for SGD) */
  void remove(char* kept_flag, int num_kept) override;
  
  /** @brief Duplicate optimizer state (no-op for SGD) */
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  
  /** @brief Reset optimizer state for specific gaussians (no-op for SGD) */
  void reset(int* indices, int num_reset) override;
  
  /** @brief Reset opacity-related optimizer state (no-op for SGD) */
  void reset_opacity() override;

private:
  SGDParameters m_sgd_params;
};

}  // namespace tinygs