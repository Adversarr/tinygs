#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct AosGaussianAdam {
  vec3 mean;                                     // 12
  float opacity;                                 // 16
  vec4 rotation;                                 // 32
  alignas(16) vec3 scale;                        // 44
  vec3 shs[kMaxSphericalHarmonicsCoefficients];  // 240
  uint32_t count;                                // 244
  uint32_t padding[3];                           // 256
};

struct AdamWParameters {
  /// Shared parameters
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;

  /// AdaBound extension (from tiny-cuda-nn)
  bool enable_adabound = false;

  /// @brief Default constructor with default values
  AdamWParameters() = default;

  /// @brief Construct from JSON configuration
  explicit AdamWParameters(const json& config);

  /// @brief Convert parameters to JSON
  json to_json() const;

  /// @brief Load parameters from JSON
  void from_json(const json& config);
};

/// @brief AdamW optimizer implementation for Gaussian Splatting
class AdamW final : public OptimizerBase {
public:
  /// @brief Construct AdamW optimizer
  AdamW(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~AdamW() override = default;

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
  AdamWParameters m_adam_params;
  // thrust::device_vector<AosGaussianAdam> m_gaussians_adam;
  std::unique_ptr<GPUBuffer<AosGaussianAdam>> m_gaussians_adam;

  // /// Moment estimates for each parameter type
  // thrust::device_vector<vec3> m_means_first_second;
  // thrust::device_vector<float> m_opacities_first_second;
  // thrust::device_vector<vec4> m_rotations_first_second;
  // thrust::device_vector<vec3> m_scales_first_second;
  // thrust::device_vector<vec3> m_sh_coefficient_0_first_second;
  // thrust::device_vector<vec3> m_sh_coefficients_rest_first_second;
  // thrust::device_vector<uint32_t> m_gaussian_steps;
};

}  // namespace tinygs
