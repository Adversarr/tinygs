#include "tinygs/strategy/strategy.hpp"
#include "tinygs/strategy/default.hpp"
#include "tinygs/strategy/mcmc.hpp"
#include <algorithm>

namespace tinygs {

StrategyBase::StrategyBase(std::shared_ptr<GPUGaussian3d> gaussians,
                           std::shared_ptr<GPUGaussian3d> gaussians_grad,
                           std::shared_ptr<OptimizerBase> optimizer)
  : m_gaussians(gaussians), m_gaussians_grad(gaussians_grad), m_optimizer(optimizer) {
}

void StrategyBase::step(const RasterizeContext& ctx) {
  step_impl(ctx);
  ++m_step_count;
}

void StrategyBase::on_remove(char* kept_flag, int num_kept) {
  if (num_kept <= 0) return;

  if (m_gaussians) {
    m_gaussians->remove(kept_flag, num_kept);
  }
  if (m_gaussians_grad) {
    m_gaussians_grad->remove(kept_flag, num_kept);
  }
  if (m_optimizer) {
    m_optimizer->remove(kept_flag, num_kept);
  }
}

void StrategyBase::on_duplicate(int* indices, int* new_indices, int num_duplications) {
  if (num_duplications <= 0) return;
  
  if (m_gaussians) {
    m_gaussians->append(num_duplications);
  }
  if (m_gaussians_grad) {
    m_gaussians_grad->append(num_duplications);
  }
  if (m_optimizer) {
    m_optimizer->duplicate(indices, new_indices, num_duplications);
  }
}

void StrategyBase::on_reset(int* indices, int num_reset) {
  if (num_reset <= 0) return;
  
  if (m_optimizer) {
    m_optimizer->reset(indices, num_reset);
  }
}

void StrategyBase::on_reset_opacity() {
  if (m_optimizer) {
    m_optimizer->reset_opacity();
  }
}

void StrategyBase::set_params(const json& config) {
  m_params.from_json(config);
}

json StrategyBase::get_params() const {
  return m_params.to_json();
}

json StrategyParams::to_json() const {
  json j;
  j["pruning_opacity_threshold"] = pruning_opacity_threshold;
  j["pruning_scale_threshold"] = pruning_scale_threshold;
  j["max_screen_size"] = max_screen_size;
  j["duplicate_grad_threshold"] = duplicate_grad_threshold;
  j["duplicate_scale_threshold"] = duplicate_scale_threshold;
  j["refine_every"] = refine_every;
  j["start_refine"] = start_refine;
  j["end_refine"] = end_refine;
  j["max_num_gaussians"] = max_num_gaussians;
  j["reset_every"] = reset_every;
  j["seed"] = seed;
  return j;
}

void StrategyParams::from_json(const json& config) {
  if (config.contains("pruning_opacity_threshold")) {
    pruning_opacity_threshold = config["pruning_opacity_threshold"].get<float>();
  }
  if (config.contains("pruning_scale_threshold")) {
    pruning_scale_threshold = config["pruning_scale_threshold"].get<float>();
  }
  if (config.contains("max_screen_size")) {
    max_screen_size = config["max_screen_size"].get<int>();
  }
  if (config.contains("duplicate_grad_threshold")) {
    duplicate_grad_threshold = config["duplicate_grad_threshold"].get<float>();
  }
  if (config.contains("duplicate_scale_threshold")) {
    duplicate_scale_threshold = config["duplicate_scale_threshold"].get<float>();
  }
  if (config.contains("refine_every")) {
    refine_every = config["refine_every"].get<int>();
  }
  if (config.contains("start_refine")) {
    start_refine = config["start_refine"].get<int>();
  }
  if (config.contains("end_refine")) {
    end_refine = config["end_refine"].get<int>();
  }
  if (config.contains("max_num_gaussians")) {
    max_num_gaussians = config["max_num_gaussians"].get<int>();
  }
  if (config.contains("reset_every")) {
    reset_every = config["reset_every"].get<int>();
  }
  if (config.contains("seed")) {
    seed = config["seed"].get<uint64_t>();
  }
}

StrategyParams::StrategyParams(const json& config) {
  from_json(config);
}

std::unique_ptr<StrategyBase> create_strategy(const std::string& strategy_type,
                                             std::shared_ptr<GPUGaussian3d> gaussians,
                                             std::shared_ptr<GPUGaussian3d> gaussians_grad,
                                             std::shared_ptr<OptimizerBase> optimizer) {
  std::string lower_strategy_type = strategy_type;
  std::transform(lower_strategy_type.begin(), lower_strategy_type.end(), lower_strategy_type.begin(), ::tolower);
  
  if (lower_strategy_type == "default") {
    return std::make_unique<DefaultStrategy>(gaussians, gaussians_grad, optimizer);
  } else if (lower_strategy_type == "mcmc") {
    return std::make_unique<MCMCStrategy>(gaussians, gaussians_grad, optimizer);
  } else {
    throw std::runtime_error("Unknown strategy type: " + strategy_type);
  }
}

} // namespace tinygs