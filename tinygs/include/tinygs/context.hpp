#pragma once

#include <memory>

#include "tinygs/core/model.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/trainer/trainer.hpp"

namespace tinygs {

class Context {
public:
  std::shared_ptr<OptimizerBase> m_optimizer;       // The optimizer for Gaussians.
  std::shared_ptr<GaussianModel> m_gaussian_model;  // 3D gaussian splatting parameters.
  std::shared_ptr<Trainer> m_trainer;
};

}  // namespace tinygs