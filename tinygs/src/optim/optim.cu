#include "tinygs/optim/optim.hpp"

namespace tinygs {

OptimizerBase::OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
    m_gaussians(gaussians), m_gaussians_grad(gaussians_grad) {
}

void OptimizerBase::set_lr(float new_lr) {
  m_global_lr = new_lr;
}

float OptimizerBase::get_lr() const {
  return m_global_lr;
}

} // namespace tinygs