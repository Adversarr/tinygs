#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct SimpleAdamParameters {
  /// Shared parameters
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;

  /// @brief Default constructor with default values
  SimpleAdamParameters() = default;

  /// @brief Construct from JSON configuration
  explicit SimpleAdamParameters(const json& config);

  /// @brief Convert parameters to JSON
  json to_json() const;

  /// @brief Load parameters from JSON
  void from_json(const json& config);
};

/// @brief SimpleAdam optimizer implementation for Gaussian Splatting
class SimpleAdam final : public OptimizerBase {
public:
  /// @brief Construct SimpleAdam optimizer
  SimpleAdam(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~SimpleAdam() override = default;

  /// @brief Reset all optimizer state
  void reset() override;
  
  /// @brief Perform one optimization step
  void step(float scale) override;
  
  /// @brief Remove optimizer state for flagged gaussians
  void remove(char* kept_flag, int num_kept) override;
  
  /// @brief Duplicate optimizer state for new gaussians
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  
  /// @brief Reset optimizer state for specific gaussians
  void reset(int* indices, int num_reset) override;
  
  /// @brief Reset opacity-related optimizer state
  void reset_opacity() override;
  
  /// @brief Set optimizer parameters from JSON
  void set_params(const json& config) override;
  
  /// @brief Get optimizer parameters as JSON
  json get_params() const override;

private:
  SimpleAdamParameters m_adam_params;
  uint32_t m_global_steps = 0;

  /// First moment estimates for each parameter type
  thrust::device_vector<vec3> m_means_first;
  thrust::device_vector<float> m_opacities_first;
  thrust::device_vector<vec4> m_rotations_first;
  thrust::device_vector<vec3> m_scales_first;
  thrust::device_vector<vec3> m_sh_coefficient_0_first;
  thrust::device_vector<vec3> m_sh_coefficients_rest_first;

  /// Second moment estimates for each parameter type
  thrust::device_vector<vec3> m_means_second;
  thrust::device_vector<float> m_opacities_second;
  thrust::device_vector<vec4> m_rotations_second;
  thrust::device_vector<vec3> m_scales_second;
  thrust::device_vector<vec3> m_sh_coefficient_0_second;
  thrust::device_vector<vec3> m_sh_coefficients_rest_second;
};

}  // namespace tinygs
