#pragma once

#include "tinygs/rasterizer/rasterizer.hpp"

namespace tinygs {

class Gsplat : public RasterizerBase {
public:

  ~Gsplat() override = default;

  void forward(const RasterizeContext& params) override;

  void backward(const RasterizeContext& params) override;

  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) override;

private:
};

}