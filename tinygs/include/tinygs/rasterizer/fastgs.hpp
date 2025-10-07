#pragma once
#include "tinygs/rasterizer/rasterizer.hpp"

namespace tinygs {

class FastGSRasterizer final : public RasterizerBase {
public:
  FastGSRasterizer();

  ~FastGSRasterizer() override;

  void forward(const RasterizeContext& ctx) override;

  void backward(RasterizeContext& ctx) override;

  void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians) override;

  /// @brief Get current rasterizer parameters as JSON
  json get_params() const override;

  /// @brief Set rasterizer parameters from JSON configuration
  void set_params(const json& params) override;

  struct Impl;
private:
  std::unique_ptr<Impl> m_impl;
  bool absgrad = false;  /// @brief Whether to use absolute gradient for densification.
};

}