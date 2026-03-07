#pragma once
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"
#include <memory>

namespace tinygs {

struct AdamParameters {
  /// Shared parameters
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;
  bool decouple_decay = false; // AdamW support
  float weight_decay = 0.0f;   // Standard AdamW decoupled weight decay coefficient
  bool tf_style = false;       // false = PyTorch style (sqrt(v̂_t) + ε), true = TensorFlow style (sqrt(v̂_t + ε))
  bool copy_state_on_duplicate = false;  // true: copy optimizer state from source; false: zero state for new gaussians

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
  Adam(BackendRuntime& runtime,
       std::shared_ptr<GPUGaussian3d> gaussians,
       std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~Adam() override;

  /// @brief Reset all optimizer state
  void reset(BackendQueue* queue) override;
  
  /// @brief Perform one optimization step
  void step(float scale, const BackendQueue* queue) override;
  void step(const GroupStepConfig& step_config, const BackendQueue* queue) override;
  
  /// @brief Remove optimizer state for flagged gaussians
  void remove(char* kept_flag, int num_kept, BackendQueue* queue) override;
  
  /// @brief Duplicate optimizer state for new gaussians
  void duplicate(int* indices, int* new_indices, int num_duplicate, BackendQueue* queue) override;
  
  /// @brief Reset optimizer state for specific gaussians
  void reset(int* indices, int num_reset) override;
  
  /// @brief Reset opacity-related optimizer state
  void reset_opacity(BackendQueue* queue) override;
  
  /// @brief Reorder Gaussians based on provided indices
  void reorder(uint* indices, BackendQueue* queue) override;
  
  /// @brief Set optimizer parameters from JSON
  void set_params(const json& config) override;
  
  /// @brief Get optimizer parameters as JSON
  json get_params() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  void step_adam(float scale, const BackendQueue* queue);
  void step_adamw(float scale, const BackendQueue* queue);

  AdamParameters m_adam_params;
  uint32_t m_global_steps = 0;
  uint32_t m_means_steps = 0;
  uint32_t m_shs_steps = 0;
  uint32_t m_opacities_steps = 0;
  uint32_t m_scales_steps = 0;
  uint32_t m_rotations_steps = 0;
};

}  // namespace tinygs
