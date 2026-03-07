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
  AdamPerGaussian(BackendRuntime& runtime,
                  std::shared_ptr<GPUGaussian3d> gaussians,
                  std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~AdamPerGaussian() override;

  void reset(BackendQueue* queue) override;

  void step(float scale, const BackendQueue* queue) override;
  void step(const GroupStepConfig& step_config, const BackendQueue* queue) override;

  void remove(char* kept_flag, int num_kept, BackendQueue* queue) override;
  void duplicate(int* indices, int* new_indices, int num_duplicate, BackendQueue* queue) override;
  void reset(int* indices, int num_reset) override;
  void reset_opacity(BackendQueue* queue) override;
  void reorder(uint* indices, BackendQueue* queue) override;

  void set_params(const json& config) override;
  json get_params() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  void step_adam(float scale, const BackendQueue* queue);
  void step_adamw(float scale, const BackendQueue* queue);

  AdamParameters m_adam_params;
};

}  // namespace tinygs
