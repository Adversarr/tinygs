#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct LionParameters {
  /// Coefficients for update moving average and gradient moving average
  float beta1 = 0.9f;   // used in the update direction mix
  float beta2 = 0.99f;  // momentum tracking

  /// @brief Default constructor with default values
  LionParameters() = default;

  /// @brief Construct from JSON configuration
  explicit LionParameters(const json& config) { from_json(config); }

  /// @brief Convert parameters to JSON
  json to_json() const {
    json j;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    return j;
  }

  /// @brief Load parameters from JSON
  void from_json(const json& config) {
    if (config.contains("beta1")) beta1 = config.at("beta1").get<float>();
    if (config.contains("beta2")) beta2 = config.at("beta2").get<float>();
  }
};

/// @brief LION optimizer (Evolved Sign Momentum) for Gaussian Splatting
class Lion final : public OptimizerBase {
public:
  /// @brief Construct Lion optimizer
  Lion(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~Lion() override = default;

  /// @brief Reset all optimizer state
  void reset() override;

  /// @brief Perform one optimization step
  void step(float scale, cudaStream_t stream) override;

  /// @brief Remove optimizer state for flagged gaussians
  void remove(char* kept_flag, int num_kept) override;

  /// @brief Duplicate optimizer state for new gaussians
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;

  /// @brief Reset optimizer state for specific gaussians
  void reset(int* indices, int num_reset) override;

  /// @brief Reset opacity-related optimizer state
  void reset_opacity() override;

  /// @brief Reorder gaussians according to the indices
  void reorder(uint* indices) override;

  /// @brief Set optimizer parameters from JSON
  void set_params(const json& config) override;

  /// @brief Get optimizer parameters as JSON
  json get_params() const override;

private:
  LionParameters m_lion_params;

  /// Momentum (first moment) buffers for each parameter type
  thrust::device_vector<vec3> m_means_m;
  thrust::device_vector<float> m_opacities_m;
  thrust::device_vector<vec4> m_rotations_m;
  thrust::device_vector<vec3> m_scales_m;
  thrust::device_vector<vec3> m_sh_coefficient_0_m;
  thrust::device_vector<vec3> m_sh_coefficients_rest_m;
};

}  // namespace tinygs