#include "tinygs/rasterizer/rasterizer.hpp"

namespace tinygs {

RasterizerBase::RasterizerBase() {
  m_memory_arena = std::make_shared<GPUMemoryArena>();
}

void RasterizerBase::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) {
  m_gaussians = gaussians;
}

}
