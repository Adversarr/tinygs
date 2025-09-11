#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct SGDParameters {
  // No momentum parameters needed for basic SGD

  /** @brief Default constructor with default values */
  SGDParameters() = default;

  /** @brief Construct from JSON configuration */
  explicit SGDParameters(const json& config);

  /** @brief Convert parameters to JSON */
  json to_json() const;

  /** @brief Load parameters from JSON */
  void from_json(const json& config);
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
   */
  SGD(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

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

  /**
   * @brief Set optimizer parameters from JSON configuration
   * 
   * @param config JSON configuration containing optimizer parameters
   */
  void set_params(const json& config) override;

  /**
   * @brief Get current optimizer parameters as JSON
   * 
   * @return JSON object containing current optimizer parameters
   */
  json get_params() const override;

private:
  SGDParameters m_sgd_params;
};

}  // namespace tinygs