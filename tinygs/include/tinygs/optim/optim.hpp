#pragma once

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

/// @brief Per-parameter-group learning rates and regularization for Gaussian optimization.
///
/// Each Gaussian has five parameter groups (means, SH coefficients, opacities,
/// scales, rotations).  The final effective LR is `group_lr * global_lr` where
/// `global_lr` is set by the LR scheduler via `OptimizerBase::set_lr()`.
struct GaussianOptimizationParams {
  /// Absolute gradient clipping threshold (L-inf per element).
  /// Set to 0.0 to disable clipping.
  float max_grad_1 = 1.0f;
  /// When true, skip the optimizer update for elements whose gradient is exactly zero.
  bool skip_zero_grad = false;

  /// Per-group learning rates
  float means_lr = 1.6e-4f;
  float shs_lr = 2.5e-3f;
  float opacities_lr = 5.0e-2f;
  float scales_lr = 5.0e-3f;
  float rotations_lr = 1.0e-3f;

  /// L1 regularization coefficients (added to gradient before step)
  float opacities_l1 = 0.0f;
  float scales_l1 = 0.0f;

  GaussianOptimizationParams() = default;
  explicit GaussianOptimizationParams(const json& config);

  json to_json() const;
  void from_json(const json& config);
};

/// @brief Abstract base class for optimizers that update GPUGaussian3d parameters.
///
/// Contract:
///   - Constructed with shared pointers to the Gaussian data and its gradient buffer.
///   - `step(scale, stream)` reads gradients, updates parameters, and clears grad state
///     internally.  `scale` includes both inverse-grad-scaler and 1/accumulation_steps.
///   - `remove()` / `duplicate()` / `reorder()` update internal momentum buffers to
///     stay consistent after the Strategy adds or removes Gaussians.
///   - All CUDA work is enqueued on the provided `stream`.
///
/// Implementations: "adam" (Adam), "adamw" (AdamW), "sgd" (vanilla SGD).
class OptimizerBase {
public:
  OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  virtual ~OptimizerBase() = default;

  /// @brief Set the global learning rate multiplier (called by LrScheduler).
  void set_lr(float new_lr);

  /// @brief Get the current global learning rate multiplier.
  float get_lr() const;

  virtual void set_params(const json& config);
  virtual json get_params() const;

  /// @brief Perform one optimization step.
  /// @param scale Combined scale factor: `(1 / grad_scaler) / accumulate_grad_steps`.
  /// @param stream CUDA stream for all kernels.
  virtual void step(float scale, cudaStream_t stream) = 0;

  /// @brief Reset all internal momentum / variance buffers (e.g. after reinit).
  virtual void reset();

  /// @brief Remove pruned Gaussians and shrink momentum buffers.
  /// @param kept_flag Per-Gaussian flag (1 = keep, 0 = remove); device memory.
  /// @param num_kept Total number of Gaussians after removal.
  virtual void remove(char* kept_flag, int num_kept){}

  /// @brief Expand momentum buffers after Gaussian duplication.
  /// @param indices Source indices for duplicated Gaussians; device memory.
  /// @param new_indices Destination indices in the expanded buffer; device memory.
  /// @param num_duplicate Number of new Gaussians added.
  virtual void duplicate(int* indices, int* new_indices, int num_duplicate){}

  /// @brief Zero out momentum buffers for specific Gaussians (e.g. after split).
  virtual void reset(int* indices, int num_reset) = 0;

  /// @brief Reorder internal buffers to match a new Gaussian ordering.
  ///        Performs a gather: new[i] = old[indices[i]].
  /// @param indices New-to-old index mapping; device memory, length = num_gaussians.
  virtual void reorder(uint* indices) = 0;

  /// @brief Reset opacity values (implementation-specific clamping / reinit).
  virtual void reset_opacity() = 0;

  /// @brief Re-bind Gaussian pointers and reset all state.
  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  std::shared_ptr<GPUGaussian3d> get_gaussians() const { return m_gaussians; }
  std::shared_ptr<GPUGaussian3d> get_gaussians_grad() const { return m_gaussians_grad; }

  GaussianOptimizationParams get_optimization_params() const noexcept { return m_params; }

protected:
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUGaussian3d> m_gaussians_grad;

  float m_global_lr = 1.0f;
  GaussianOptimizationParams m_params;
};

/// @brief Factory: create an optimizer by type name.
/// @param optimizer_type One of: "adam", "adamw", "sgd".
/// @param gaussians Gaussian parameters to optimize.
/// @param gaussians_grad Gradient buffer (same layout as gaussians).
std::unique_ptr<OptimizerBase> create_optimizer(const std::string& optimizer_type,
                                                std::shared_ptr<GPUGaussian3d> gaussians,
                                                std::shared_ptr<GPUGaussian3d> gaussians_grad);

}  // namespace tinygs