#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct AdamParameters {
  /// Shared parameters
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;
  bool decouple_decay = false; // AdamW support
  std::string decay_reduction = "mean"; // "mean" or "sum"

  /// @brief Default constructor with default values
  AdamParameters() = default;

  /// @brief Construct from JSON configuration
  explicit AdamParameters(const json& config);

  /// @brief Convert parameters to JSON
  json to_json() const;

  /// @brief Load parameters from JSON
  void from_json(const json& config);
};

/// @brief Adam optimizer implementation for Gaussian Splatting
class Adam final : public OptimizerBase {
public:
  /// @brief Construct Adam optimizer
  Adam(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~Adam() override = default;

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
  
  /// @brief Reorder Gaussians based on provided indices
  void reorder(uint* indices) override;
  
  /// @brief Set optimizer parameters from JSON
  void set_params(const json& config) override;
  
  /// @brief Get optimizer parameters as JSON
  json get_params() const override;

private:

  void step_adam(float scale, cudaStream_t stream);
  void step_adamw(float scale, cudaStream_t stream);

  AdamParameters m_adam_params;
  uint32_t m_global_steps = 0;

  /// First moment estimates for each parameter type
  thrust::device_vector<vec3> m_means_first;
  thrust::device_vector<float> m_opacities_first;
  thrust::device_vector<vec4> m_rotations_first;
  thrust::device_vector<vec3> m_scales_first;
  /// Per-degree SH first moments (flat float SoA, same layout as GPUGaussian3d SH buffers).
  thrust::device_vector<float> m_sh0_first;   ///< size = 3 * N
  thrust::device_vector<float> m_sh1_first;   ///< size = 9 * N
  thrust::device_vector<float> m_sh2_first;   ///< size = 15 * N
  thrust::device_vector<float> m_sh3_first;   ///< size = 21 * N

  /// Second moment estimates for each parameter type
  thrust::device_vector<vec3> m_means_second;
  thrust::device_vector<float> m_opacities_second;
  thrust::device_vector<vec4> m_rotations_second;
  thrust::device_vector<vec3> m_scales_second;
  /// Per-degree SH second moments (flat float SoA, same layout as GPUGaussian3d SH buffers).
  thrust::device_vector<float> m_sh0_second;  ///< size = 3 * N
  thrust::device_vector<float> m_sh1_second;  ///< size = 9 * N
  thrust::device_vector<float> m_sh2_second;  ///< size = 15 * N
  thrust::device_vector<float> m_sh3_second;  ///< size = 21 * N
};

}  // namespace tinygs
