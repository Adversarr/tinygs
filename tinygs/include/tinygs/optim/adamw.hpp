#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct AdamWParameters {
  // Shared parameters
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;

  /// NOTE: from tiny-cuda-nn Adam implementation
  // AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
  bool enable_adabound = false;

  /** @brief Default constructor with default values */
  AdamWParameters() = default;

  /** @brief Construct from JSON configuration */
  explicit AdamWParameters(const json& config);

  /** @brief Convert parameters to JSON */
  json to_json() const;

  /** @brief Load parameters from JSON */
  void from_json(const json& config);
};

/**
 * @brief AdamW optimizer implementation for Gaussian Splatting with adaptive moment estimation
 * 
 * This optimizer maintains first and second moment estimates for each parameter and applies
 * bias correction. It supports gradient scaling separate from learning rate scaling.
 */
class AdamW final : public OptimizerBase {
public:
  /**
   * @brief Construct AdamW optimizer with specified parameters
   * @param gaussians Shared pointer to GPU Gaussian parameters
   * @param gaussians_grad Shared pointer to GPU Gaussian gradients
   * @param params AdamW-specific parameters (beta1, beta2, epsilon, etc.)
   */
  AdamW(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~AdamW() override = default;

  /** @brief Reset all optimizer state (moments and step counts) to zero */
  void reset() override;
  
  /** @brief Perform one optimization step with gradient scaling */
  void step(float scale) override;
  
  /** @brief Remove optimizer state for flagged gaussians */
  void remove(char* kept_flag, int num_kept) override;
  
  /** @brief Duplicate optimizer state for new gaussians */
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  
  /** @brief Reset optimizer state for specific gaussians */
  void reset(int* indices, int num_reset) override;
  
  /** @brief Reset only opacity-related optimizer state */
  void reset_opacity() override;
  
  /** @brief Set optimizer parameters from JSON configuration */
  void set_params(const json& config) override;
  
  /** @brief Get optimizer parameters as JSON configuration */
  json get_params() const override;

private:
  AdamWParameters m_adam_params;

  // TODO: replace with ours.
  thrust::device_vector<vec3> m_means_first_second;
  thrust::device_vector<float> m_opacities_first_second;
  thrust::device_vector<vec4> m_rotations_first_second;
  thrust::device_vector<vec3> m_scales_first_second;
  thrust::device_vector<vec3> m_sh_coefficient_0_first_second;
  thrust::device_vector<vec3> m_sh_coefficients_rest_first_second;
  thrust::device_vector<uint32_t> m_gaussian_steps;
};

}  // namespace tinygs
