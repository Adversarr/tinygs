#pragma once

#include <memory>
#include "tinygs/optim/optim.hpp"

namespace tinygs {

/**
 * @brief Abstract base class for learning rate schedulers
 * 
 * Provides interface for updating learning rates during training.
 * Schedulers can implement different decay strategies like constant,
 * exponential, step-wise, or cosine annealing.
 * Follows PyTorch design pattern where scheduler holds optimizer reference.
 */
class LrSchedulerBase {
public:
  /// @brief Construct scheduler with optimizer and initial learning rate
  LrSchedulerBase(const std::shared_ptr<OptimizerBase> &optimizer,
                  OptimParamGroup group,
                  float initial_lr);

  virtual ~LrSchedulerBase() = default;

  /// @brief Perform one scheduler step and update optimizer's learning rate
  virtual float step() = 0;

  /// @brief Reset scheduler to initial state
  virtual void reset() = 0;

  /// @brief Get current learning rate
  float get_lr() const;

  /// @brief Get scheduler parameters as JSON
  virtual json get_params() const = 0;

  /// @brief Set scheduler parameters from JSON
  virtual void set_params(const json& params) = 0;

protected:
  /// @brief Update both internal state and optimizer's learning rate
  void update_lr(float new_lr);

  std::shared_ptr<OptimizerBase> m_optimizer; ///< Optimizer whose learning rate is controlled
  OptimParamGroup m_group; ///< Target optimizer parameter group
  float m_current_lr; ///< Current learning rate value
};

/**
 * @brief Constant learning rate scheduler
 * 
 * Maintains the same learning rate throughout training.
 * Useful as a baseline or when no learning rate decay is desired.
 */
class ConstantLR : public LrSchedulerBase {
public:
  /// @brief Construct constant learning rate scheduler
  explicit ConstantLR(const std::shared_ptr<OptimizerBase> &optimizer,
                      OptimParamGroup group,
                      float lr=1.0f);

  /// @brief Return the constant learning rate
  float step() override;

  /// @brief Reset scheduler (no-op for constant scheduler)
  void reset() override;

  /// @brief Get scheduler parameters as JSON
  json get_params() const override;

  /// @brief Set scheduler parameters from JSON
  void set_params(const json& params) override;
};

/**
 * @brief Exponential learning rate decay scheduler
 * 
 * Applies exponential decay: lr = initial_lr * (decay_rate ^ step_count)
 * Common choice for training neural networks with gradual learning rate reduction.
 */
class ExponentialLR : public LrSchedulerBase {
public:
  /// @brief Construct exponential decay scheduler
  explicit ExponentialLR(const std::shared_ptr<OptimizerBase> &optimizer,
                         OptimParamGroup group,
                         float initial_lr = 1.0, float decay_rate = 0.999769f);

  /// @brief Apply exponential decay and return new learning rate
  float step() override;

  /// @brief Reset scheduler to initial state
  void reset() override;

  /// @brief Get scheduler parameters as JSON
  json get_params() const override;

  /// @brief Set scheduler parameters from JSON
  void set_params(const json& params) override;

private:
  float m_initial_lr;   ///< Initial learning rate value
  float m_decay_rate;   ///< Decay factor applied each step
  int m_step_count;     ///< Current step count for decay calculation
};

/// @brief Create learning rate scheduler object
/// @param scheduler_type Type of scheduler to create
/// @param optimizer Optimizer whose learning rate will be controlled
std::unique_ptr<LrSchedulerBase> create_lr_scheduler(const std::string& scheduler_type,
                                                     const std::shared_ptr<OptimizerBase>& optimizer,
                                                     OptimParamGroup group);

} // namespace tinygs