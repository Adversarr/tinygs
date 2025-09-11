#include "tinygs/optim/lr_scheduler.hpp"

#include <cmath>
#include <stdexcept>

#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

// LrSchedulerBase implementation
LrSchedulerBase::LrSchedulerBase(const std::shared_ptr<OptimizerBase> &optimizer, float initial_lr)
  : m_optimizer(optimizer), m_current_lr(initial_lr) {
  if (m_optimizer) {
    m_optimizer->set_lr(initial_lr);
  }
}

void LrSchedulerBase::update_lr(float new_lr) {
  m_current_lr = new_lr;
  if (m_optimizer) {
    m_optimizer->set_lr(new_lr);
  } else {
    log_warning("No optimizer set.");
  }
}

float LrSchedulerBase::get_lr() const {
  return m_current_lr;
}

// ConstantLR implementation
ConstantLR::ConstantLR(const std::shared_ptr<OptimizerBase> &optimizer, float lr)
  : LrSchedulerBase(optimizer, lr) {}

float ConstantLR::step() {
  // No change needed for constant LR, but ensure optimizer is updated
  update_lr(m_current_lr);
  return m_current_lr;
}

void ConstantLR::reset() {
  // No state to reset for constant scheduler
}

// ExponentialLR implementation
ExponentialLR::ExponentialLR(const std::shared_ptr<OptimizerBase> &optimizer,
                             float initial_lr, float decay_rate)
  : LrSchedulerBase(optimizer, initial_lr), m_initial_lr(initial_lr),
    m_decay_rate(decay_rate), m_step_count(0) {}

float ExponentialLR::step() {
  float new_lr = m_initial_lr * std::pow(m_decay_rate, m_step_count);
  m_step_count++;
  update_lr(new_lr);
  return new_lr;
}

void ExponentialLR::reset() {
  m_step_count = 0;
  update_lr(m_initial_lr);
}

// Factory function implementation
std::unique_ptr<LrSchedulerBase> create_lr_scheduler(const std::string& scheduler_type,
                                                     const std::shared_ptr<OptimizerBase>& optimizer) {
  if (scheduler_type == "constant") {
    return std::make_unique<ConstantLR>(optimizer);
  } else if (scheduler_type == "exponential") {
    return std::make_unique<ExponentialLR>(optimizer);
  } else {
    throw std::invalid_argument("Unknown scheduler type: " + scheduler_type);
  }
}

} // namespace tinygs