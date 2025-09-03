#include "tinygs/optim/optim.hpp"

namespace tinygs {

OptimizerBase::OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad) :
    m_gaussians(gaussians), m_gaussians_grad(gaussians_grad) {
}

}  // namespace tinygs