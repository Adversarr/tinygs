#pragma once

#include <cmath>
#include <memory>
#include "optim.hpp"

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
  /**
   * @brief Construct scheduler with optimizer and initial learning rate
   * @param optimizer Shared pointer to optimizer whose learning rate will be controlled
   * @param initial_lr Initial learning rate value
   */
  LrSchedulerBase(const std::shared_ptr<OptimizerBase> &optimizer, float initial_lr)
    : m_optimizer(optimizer), m_current_lr(initial_lr) {
    if (m_optimizer) {
      m_optimizer->set_lr(initial_lr);
    }
  }

  virtual ~LrSchedulerBase() = default;

  /**
   * @brief Perform one scheduler step and update optimizer's learning rate
   * @return Updated learning rate for the current step
   */
  virtual float step() = 0;

  /**
   * @brief Reset scheduler to initial state
   */
  virtual void reset() = 0;

  /**
   * @brief Get current learning rate
   * @return Current learning rate value
   */
  float get_lr() const { return m_current_lr; }

protected:
  /**
   * @brief Update both internal state and optimizer's learning rate
   * @param new_lr New learning rate to set
   */
  void update_lr(float new_lr) {
    m_current_lr = new_lr;
    if (m_optimizer) {
      m_optimizer->set_lr(new_lr);
    } else {
      log_warning("No optimizer set.");
    }
  }

  std::shared_ptr<OptimizerBase> m_optimizer; ///< Optimizer whose learning rate is controlled
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
  /**
   * @brief Construct constant learning rate scheduler
   * @param optimizer Shared pointer to optimizer whose learning rate will be controlled
   * @param lr Learning rate to maintain constant
   */
  explicit ConstantLR(const std::shared_ptr<OptimizerBase> &optimizer, float lr=1.0f)
    : LrSchedulerBase(optimizer, lr) {}

  /**
   * @brief Return the constant learning rate
   * @return Unchanged learning rate
   */
  float step() override {
    // No change needed for constant LR, but ensure optimizer is updated
    update_lr(m_current_lr);
    return m_current_lr;
  }

  /**
   * @brief Reset scheduler (no-op for constant scheduler)
   */
  void reset() override {
    // No state to reset for constant scheduler
  }
};

/**
 * @brief Exponential learning rate decay scheduler
 * 
 * Applies exponential decay: lr = initial_lr * (decay_rate ^ step_count)
 * Common choice for training neural networks with gradual learning rate reduction.
 */
class ExponentialLR : public LrSchedulerBase {
public:
  /**
   * @brief Construct exponential decay scheduler
   * @param optimizer Shared pointer to optimizer whose learning rate will be controlled
   * @param initial_lr Initial learning rate
   * @param decay_rate Decay factor applied each step (typically 0.9-0.99)
   */
  explicit ExponentialLR(const std::shared_ptr<OptimizerBase> &optimizer,
                         float initial_lr = 1.0, float decay_rate = 0.999769f)
    : LrSchedulerBase(optimizer, initial_lr), m_initial_lr(initial_lr),
      m_decay_rate(decay_rate), m_step_count(0) {}

  /**
   * @brief Apply exponential decay and return new learning rate
   * @return Learning rate after exponential decay
   */
  float step() override {
    float new_lr = m_initial_lr * std::pow(m_decay_rate, m_step_count);
    m_step_count++;
    update_lr(new_lr);
    return new_lr;
  }

  /**
   * @brief Reset scheduler to initial state
   */
  void reset() override {
    m_step_count = 0;
    update_lr(m_initial_lr);
  }

private:
  float m_initial_lr;   ///< Initial learning rate value
  float m_decay_rate;   ///< Decay factor applied each step
  int m_step_count;     ///< Current step count for decay calculation
};

} // namespace tinygs