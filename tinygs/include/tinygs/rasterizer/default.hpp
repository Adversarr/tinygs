#pragma once

#include "tinygs/rasterizer/rasterizer.hpp"

namespace tinygs {

class DefaultRasterizer : public RasterizerBase {
public:
  DefaultRasterizer();

  ~DefaultRasterizer() override;

  void forward(const RasterizeContext& ctx) override;

  void backward(const RasterizeContext& ctx) override;

  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) override;

  struct Impl;

private:

  std::unique_ptr<Impl> m_impl;
};

}