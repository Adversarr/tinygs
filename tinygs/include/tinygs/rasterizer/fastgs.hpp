#pragma once
#include "tinygs/rasterizer/rasterizer.hpp"

namespace tinygs {

class FastGSRasterizer final : public RasterizerBase {
public:
  FastGSRasterizer();

  ~FastGSRasterizer() override;

  void forward(const RasterizeParamsRuntime& params) override;

  void backward(const RasterizeParamsRuntime& params) override;

  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) override;
  
  struct Impl;
private:
  std::unique_ptr<Impl> m_impl;
};

}