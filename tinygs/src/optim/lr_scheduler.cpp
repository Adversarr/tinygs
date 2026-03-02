#include "tinygs/optim/lr_scheduler.hpp"

#include <cmath>
#include <stdexcept>

#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

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
    m_final_lr(initial_lr), m_delay_mult(1.0f), m_delay_steps(0),
    m_max_steps(30000), m_decay_rate(decay_rate), m_step_count(0),
    m_use_fastgs_schedule(false) {}

float ExponentialLR::step() {
  float new_lr = 0.0f;
  if (m_use_fastgs_schedule) {
    const int step = m_step_count;
    const bool zero_lr = (m_initial_lr == 0.0f && m_final_lr == 0.0f);
    if (step < 0 || zero_lr) {
      new_lr = 0.0f;
    } else {
      float delay_rate = 1.0f;
      if (m_delay_steps > 0) {
        const float ratio = static_cast<float>(step) / static_cast<float>(m_delay_steps);
        const float clipped = std::fmin(1.0f, std::fmax(0.0f, ratio));
        delay_rate = m_delay_mult + (1.0f - m_delay_mult) * std::sin(0.5f * kPi * clipped);
      }

      const int denom = std::max(m_max_steps, 1);
      const float t = std::fmin(1.0f, std::fmax(0.0f, static_cast<float>(step) / static_cast<float>(denom)));

      const float safe_init = std::fmax(m_initial_lr, 1e-20f);
      const float safe_final = std::fmax(m_final_lr, 1e-20f);
      const float log_lerp = std::exp(std::log(safe_init) * (1.0f - t) + std::log(safe_final) * t);
      new_lr = delay_rate * log_lerp;
    }
  } else {
    new_lr = m_initial_lr * std::pow(m_decay_rate, m_step_count);
  }

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
  if (m_use_fastgs_schedule) {
    params["final_lr"] = m_final_lr;
    params["delay_mult"] = m_delay_mult;
    params["delay_steps"] = m_delay_steps;
    params["max_steps"] = m_max_steps;
    params["use_fastgs_schedule"] = m_use_fastgs_schedule;
  }
  params["decay_rate"] = m_decay_rate;
  params["step_count"] = m_step_count;
  return params;
}

void ExponentialLR::set_params(const nlohmann::json& params) {
  if (params.contains("initial_lr")) {
    m_initial_lr = params["initial_lr"].get<float>();
  }
  if (params.contains("final_lr")) {
    m_final_lr = params["final_lr"].get<float>();
  }
  if (params.contains("delay_mult")) {
    m_delay_mult = params["delay_mult"].get<float>();
  }
  if (params.contains("delay_steps")) {
    m_delay_steps = params["delay_steps"].get<int>();
  }
  if (params.contains("max_steps")) {
    m_max_steps = params["max_steps"].get<int>();
  }
  if (params.contains("position_lr_final")) {
    m_final_lr = params["position_lr_final"].get<float>();
  }
  if (params.contains("position_lr_delay_mult")) {
    m_delay_mult = params["position_lr_delay_mult"].get<float>();
  }
  if (params.contains("position_lr_delay_steps")) {
    m_delay_steps = params["position_lr_delay_steps"].get<int>();
  }
  if (params.contains("position_lr_max_steps")) {
    m_max_steps = params["position_lr_max_steps"].get<int>();
  }
  if (params.contains("use_fastgs_schedule")) {
    m_use_fastgs_schedule = params["use_fastgs_schedule"].get<bool>();
  }

  if (params.contains("decay_rate")) {
    m_decay_rate = params["decay_rate"].get<float>();
  }
  if (params.contains("step_count")) {
    m_step_count = params["step_count"].get<int>();
  }

  if (params.contains("final_lr") || params.contains("delay_mult") ||
      params.contains("delay_steps") || params.contains("max_steps") ||
      params.contains("position_lr_final") || params.contains("position_lr_delay_mult") ||
      params.contains("position_lr_delay_steps") || params.contains("position_lr_max_steps")) {
    m_use_fastgs_schedule = true;
  }

  m_delay_mult = std::fmin(1.0f, std::fmax(0.0f, m_delay_mult));
  m_delay_steps = std::max(0, m_delay_steps);
  m_max_steps = std::max(1, m_max_steps);
  
  // Recalculate current learning rate based on updated parameters
  float new_lr = 0.0f;
  if (m_use_fastgs_schedule) {
    const bool zero_lr = (m_initial_lr == 0.0f && m_final_lr == 0.0f);
    if (m_step_count < 0 || zero_lr) {
      new_lr = 0.0f;
    } else {
      float delay_rate = 1.0f;
      if (m_delay_steps > 0) {
        const float ratio = static_cast<float>(m_step_count) / static_cast<float>(m_delay_steps);
        const float clipped = std::fmin(1.0f, std::fmax(0.0f, ratio));
        delay_rate = m_delay_mult + (1.0f - m_delay_mult) * std::sin(0.5f * kPi * clipped);
      }
      const float t = std::fmin(1.0f, std::fmax(0.0f, static_cast<float>(m_step_count) / static_cast<float>(m_max_steps)));
      const float safe_init = std::fmax(m_initial_lr, 1e-20f);
      const float safe_final = std::fmax(m_final_lr, 1e-20f);
      const float log_lerp = std::exp(std::log(safe_init) * (1.0f - t) + std::log(safe_final) * t);
      new_lr = delay_rate * log_lerp;
    }
  } else {
    new_lr = m_initial_lr * std::pow(m_decay_rate, m_step_count);
  }
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