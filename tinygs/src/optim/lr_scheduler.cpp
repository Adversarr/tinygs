#include "tinygs/optim/lr_scheduler.hpp"

#include <cmath>
#include <stdexcept>

#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

// LrSchedulerBase implementation
LrSchedulerBase::LrSchedulerBase(const std::shared_ptr<OptimizerBase> &optimizer,
                                 OptimParamGroup group,
                                 float initial_lr)
  : m_optimizer(optimizer), m_group(group), m_current_lr(initial_lr) {
  if (m_optimizer) {
    m_optimizer->set_lr(m_group, initial_lr);
  }
}

void LrSchedulerBase::update_lr(float new_lr) {
  m_current_lr = new_lr;
  if (m_optimizer) {
    m_optimizer->set_lr(m_group, new_lr);
  } else {
    log_warning("No optimizer set.");
  }
}

float LrSchedulerBase::get_lr() const {
  return m_current_lr;
}

// ConstantLR implementation
ConstantLR::ConstantLR(const std::shared_ptr<OptimizerBase> &optimizer,
                       OptimParamGroup group,
                       float lr)
  : LrSchedulerBase(optimizer, group, lr) {}

float ConstantLR::step() {
  // No change needed for constant LR, but ensure optimizer is updated
  update_lr(m_current_lr);
  return m_current_lr;
}

void ConstantLR::reset() {
  // No state to reset for constant scheduler
}

nlohmann::json ConstantLR::get_params() const {
  nlohmann::json params;
  params["type"] = "constant";
  params["lr"] = m_current_lr;
  return params;
}

void ConstantLR::set_params(const nlohmann::json& params) {
  if (params.contains("lr")) {
    float new_lr = params["lr"].get<float>();
    update_lr(new_lr);
  }
}

// ExponentialLR implementation
ExponentialLR::ExponentialLR(const std::shared_ptr<OptimizerBase> &optimizer,
                             OptimParamGroup group,
                             float initial_lr, float decay_rate)
  : LrSchedulerBase(optimizer, group, initial_lr), m_initial_lr(initial_lr),
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

nlohmann::json ExponentialLR::get_params() const {
  nlohmann::json params;
  params["type"] = "exponential";
  params["initial_lr"] = m_initial_lr;
  params["decay_rate"] = m_decay_rate;
  params["step_count"] = m_step_count;
  return params;
}

void ExponentialLR::set_params(const nlohmann::json& params) {
  if (params.contains("initial_lr")) {
    m_initial_lr = params["initial_lr"].get<float>();
  }
  if (params.contains("decay_rate")) {
    m_decay_rate = params["decay_rate"].get<float>();
  }
  if (params.contains("step_count")) {
    m_step_count = params["step_count"].get<int>();
  }
  
  // Recalculate current learning rate based on updated parameters
  float new_lr = m_initial_lr * std::pow(m_decay_rate, m_step_count);
  update_lr(new_lr);
}

// Factory function implementation
std::unique_ptr<LrSchedulerBase> create_lr_scheduler(const std::string& scheduler_type,
                                                     const std::shared_ptr<OptimizerBase>& optimizer,
                                                     OptimParamGroup group) {
  std::string lower_scheduler_type = to_lower(scheduler_type);
  if (lower_scheduler_type == "constant") {
    return std::make_unique<ConstantLR>(optimizer, group);
  } else if (lower_scheduler_type == "exponential") {
    return std::make_unique<ExponentialLR>(optimizer, group);
  } else {
    throw std::invalid_argument("Unknown scheduler type: " + scheduler_type);
  }
}

} // namespace tinygs