#include "tinygs/optim/adam.hpp"
#include "tinygs/optim/adam_per_gaussian.hpp"
#include "tinygs/optim/optim.hpp"
#include <cmath>
#include <nlohmann/json.hpp>

namespace tinygs {

const char* to_string(OptimParamGroup group) {
  switch (group) {
    case OptimParamGroup::Means:
      return "means";
    case OptimParamGroup::Shs:
      return "shs";
    case OptimParamGroup::Opacities:
      return "opacities";
    case OptimParamGroup::Scales:
      return "scales";
    case OptimParamGroup::Rotations:
      return "rotations";
    default:
      return "unknown";
  }
}

OptimizerBase::OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
    m_gaussians(gaussians), m_gaussians_grad(gaussians_grad) {
}

void OptimizerBase::set_params(const json& config) {
  m_params.from_json(config);
}

json OptimizerBase::get_params() const {
  return m_params.to_json();
}

void OptimizerBase::set_lr(float new_lr) {
  m_global_lr = new_lr;
  m_means_global_lr = new_lr;
  m_shs_global_lr = new_lr;
  m_opacities_global_lr = new_lr;
  m_scales_global_lr = new_lr;
  m_rotations_global_lr = new_lr;
}

float OptimizerBase::get_lr() const {
  return m_global_lr;
}

void OptimizerBase::set_lr(OptimParamGroup group, float new_lr) {
  switch (group) {
    case OptimParamGroup::Means:
      m_means_global_lr = new_lr;
      m_global_lr = new_lr;
      break;
    case OptimParamGroup::Shs:
      m_shs_global_lr = new_lr;
      break;
    case OptimParamGroup::Opacities:
      m_opacities_global_lr = new_lr;
      break;
    case OptimParamGroup::Scales:
      m_scales_global_lr = new_lr;
      break;
    case OptimParamGroup::Rotations:
      m_rotations_global_lr = new_lr;
      break;
  }
}

float OptimizerBase::get_lr(OptimParamGroup group) const {
  switch (group) {
    case OptimParamGroup::Means:
      return m_means_global_lr;
    case OptimParamGroup::Shs:
      return m_shs_global_lr;
    case OptimParamGroup::Opacities:
      return m_opacities_global_lr;
    case OptimParamGroup::Scales:
      return m_scales_global_lr;
    case OptimParamGroup::Rotations:
      return m_rotations_global_lr;
  }
  return m_global_lr;
}

void OptimizerBase::step(const GroupStepConfig& step_config, cudaStream_t stream) {
  if (!step_config.any_update()) {
    return;
  }
  if (!(step_config.update_means && step_config.update_shs && step_config.update_opacities &&
        step_config.update_scales && step_config.update_rotations)) {
    throw std::runtime_error("This optimizer does not support selective group stepping.");
  }
  const float tol = 1e-7f;
  const float s = step_config.means_scale;
  if (std::fabs(step_config.shs_scale - s) > tol || std::fabs(step_config.opacities_scale - s) > tol ||
      std::fabs(step_config.scales_scale - s) > tol || std::fabs(step_config.rotations_scale - s) > tol) {
    throw std::runtime_error("This optimizer requires equal per-group scales when using grouped step().");
  }
  step(s, stream);
}

void OptimizerBase::reset() {
  // nothing to do
}

void OptimizerBase::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians,
                                  std::shared_ptr<GPUGaussian3d> gaussians_grad) {
  m_gaussians = gaussians;
  m_gaussians_grad = gaussians_grad;
}

json GaussianOptimizationParams::to_json() const {
  json j;
  j["max_grad_1"] = max_grad_1;
  j["skip_zero_grad"] = skip_zero_grad;
  j["means_lr"] = means_lr;
  j["shs_lr"] = shs_lr;
  j["sh1_lr_scale"] = sh1_lr_scale;
  j["sh2_lr_scale"] = sh2_lr_scale;
  j["sh3_lr_scale"] = sh3_lr_scale;
  j["opacities_lr"] = opacities_lr;
  j["scales_lr"] = scales_lr;
  j["rotations_lr"] = rotations_lr;
  j["opacities_l1"] = opacities_l1;
  j["scales_l1"] = scales_l1;
  return j;
}

void GaussianOptimizationParams::from_json(const json& config) {
  if (config.contains("max_grad_1")) {
    max_grad_1 = config["max_grad_1"];
  }
  if (config.contains("skip_zero_grad")) {
    skip_zero_grad = config["skip_zero_grad"];
  }
  if (config.contains("means_lr")) {
    means_lr = config["means_lr"];
  }
  if (config.contains("shs_lr")) {
    shs_lr = config["shs_lr"];
  }
  if (config.contains("sh1_lr_scale")) {
    sh1_lr_scale = config["sh1_lr_scale"];
  }
  if (config.contains("sh2_lr_scale")) {
    sh2_lr_scale = config["sh2_lr_scale"];
  }
  if (config.contains("sh3_lr_scale")) {
    sh3_lr_scale = config["sh3_lr_scale"];
  }
  if (config.contains("opacities_lr")) {
    opacities_lr = config["opacities_lr"];
  }
  if (config.contains("scales_lr")) {
    scales_lr = config["scales_lr"];
  }
  if (config.contains("rotations_lr")) {
    rotations_lr = config["rotations_lr"];
  }
  if (config.contains("opacities_l1")) {
    opacities_l1 = config["opacities_l1"];
  }
  if (config.contains("scales_l1")) {
    scales_l1 = config["scales_l1"];
  }
}

std::unique_ptr<OptimizerBase> create_optimizer(const std::string& optimizer_type,
                                                std::shared_ptr<GPUGaussian3d> gaussians,
                                                std::shared_ptr<GPUGaussian3d> gaussians_grad) {
  std::string lower_optimizer_type = to_lower(optimizer_type);
  if (lower_optimizer_type == "adam") {
    return std::make_unique<Adam>(gaussians, gaussians_grad);
  } else if (lower_optimizer_type == "adam_per_gaussian" || lower_optimizer_type == "adam_pg") {
    return std::make_unique<AdamPerGaussian>(gaussians, gaussians_grad);
  } else {
    throw std::runtime_error(
        "Unknown optimizer type: " + optimizer_type +
        ". Supported: adam, adam_per_gaussian. Use decouple_decay=true for AdamW mode.");
  }
}
} // namespace tinygs