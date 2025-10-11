#include "tinygs/optim/adamw.hpp"
#include "tinygs/optim/sgd.hpp"
#include "tinygs/optim/simple_adam.hpp"
#include "tinygs/optim/lion.hpp"
#include "tinygs/optim/lamb.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/optim/adan.hpp"
#include <nlohmann/json.hpp>

namespace tinygs {

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
}

float OptimizerBase::get_lr() const {
  return m_global_lr;
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
    return std::make_unique<AdamW>(gaussians, gaussians_grad);
  } else if (lower_optimizer_type == "simple_adam") {
    return std::make_unique<SimpleAdam>(gaussians, gaussians_grad);
  } else if (lower_optimizer_type == "sgd") {
    return std::make_unique<SGD>(gaussians, gaussians_grad);
  } else if (lower_optimizer_type == "lion") {
    return std::make_unique<Lion>(gaussians, gaussians_grad);
  } else if (lower_optimizer_type == "lamb") {
    return std::make_unique<Lamb>(gaussians, gaussians_grad);
  } else if (lower_optimizer_type == "adan") {
    return std::make_unique<Adan>(gaussians, gaussians_grad);
  } else {
    throw std::runtime_error("Unknown optimizer type: " + optimizer_type);
  }
}
} // namespace tinygs