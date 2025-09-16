#include "tinygs/rasterizer/rasterizer.hpp"
#include "tinygs/rasterizer/fastgs.hpp"
#include "tinygs/rasterizer/default.hpp"

namespace tinygs {

RasterizerBase::RasterizerBase() {
  // m_memory_arena = std::make_shared<GPUMemoryArena>();
}

void RasterizerBase::set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) {
  m_gaussians = gaussians;
}

std::unique_ptr<RasterizerBase> create_rasterizer(const std::string& rasterizer_type) {
  std::string lower_rasterizer_type = to_lower(rasterizer_type);
  if (lower_rasterizer_type == "default") {
    return std::make_unique<DefaultRasterizer>();
  } else if (lower_rasterizer_type == "fastgs") {
    return std::make_unique<FastGSRasterizer>();
  } else {
    throw std::runtime_error("Unknown rasterizer type: " + rasterizer_type);
  }
}

}
