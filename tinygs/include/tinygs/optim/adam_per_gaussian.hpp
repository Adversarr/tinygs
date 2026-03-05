#pragma once

#include "tinygs/common.hpp"
#include "tinygs/optim/adam.hpp"
#include "tinygs/optim/optim.hpp"
#include <memory>

namespace tinygs {

/// @brief Adam/AdamW variant with shared per-Gaussian iteration counters.
///
/// Bias correction uses a per-Gaussian step counter t. New Gaussians created by
/// duplication start with t=0 and are incremented on the first optimizer step.
class AdamPerGaussian final : public OptimizerBase {
public:
  AdamPerGaussian(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~AdamPerGaussian() override;

  void reset() override;

  void step(float scale, BackendStream stream) override;
  void step(const GroupStepConfig& step_config, BackendStream stream) override;

  void remove(char* kept_flag, int num_kept) override;
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  void reset(int* indices, int num_reset) override;
  void reset_opacity() override;
  void reorder(uint* indices) override;

  void set_params(const json& config) override;
  json get_params() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  void step_adam(float scale, BackendStream stream);
  void step_adamw(float scale, BackendStream stream);

  AdamParameters m_adam_params;
};

}  // namespace tinygs
