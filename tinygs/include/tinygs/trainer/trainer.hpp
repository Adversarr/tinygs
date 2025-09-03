#pragma once

#include "core/gpu_gaussian.hpp"
#include "loss/loss.hpp"
#include "optim/optim.hpp"
#include "rasterizer/rasterizer.hpp"

namespace tinygs {
class Trainer {
public:
  Trainer() = default;
  ~Trainer() = default;

  void forward();
  void backward();
  void optimizer_step();
  void strategy_step();

  /// @brief Launch training for num_steps steps.
  void train(int num_steps);

  // float accumulate_loss();
private:
  std::shared_ptr<GPUGaussian3d> m_model;
  std::unique_ptr<OptimizerBase> m_optimizer;
  std::unique_ptr<RasterizerBase> m_rasterizer;
  std::vector<std::unique_ptr<LossBase>> m_losses;

  size_t m_current_step = 0;
  // TODO: strategy.
};
}  // namespace tinygs