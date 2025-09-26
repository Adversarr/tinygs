#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct SGDParameters {
  /// No momentum parameters needed for basic SGD

  /// @brief Default constructor with default values
  SGDParameters() = default;

  /// @brief Construct from JSON configuration
  explicit SGDParameters(const json& config);

  /// @brief Convert parameters to JSON
  json to_json() const;

  /// @brief Load parameters from JSON
  void from_json(const json& config);
};

/// @brief Basic SGD optimizer without momentum for Gaussian Splatting
class SGD : public OptimizerBase {
public:
  /// @brief Construct SGD optimizer
  SGD(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~SGD() override = default;

  /// @brief Reset optimizer state (no-op for SGD)
  void reset() override;
  
  /// @brief Perform one optimization step
  void step(float scale, cudaStream_t stream) override;
  
  /// @brief Remove optimizer state (no-op for SGD)
  void remove(char* kept_flag, int num_kept) override;
  
  /// @brief Duplicate optimizer state (no-op for SGD)
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  
  /// @brief Reset optimizer state for specific gaussians (no-op for SGD)
  void reset(int* indices, int num_reset) override;
  
  /// @brief Reset opacity-related optimizer state (no-op for SGD)
  void reset_opacity() override;

  /// @brief Set optimizer parameters from JSON
  void set_params(const json& config) override;

  /// @brief Get current optimizer parameters as JSON
  json get_params() const override;

private:
  SGDParameters m_sgd_params;
};

}  // namespace tinygs