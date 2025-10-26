#pragma once
#include "tinygs/rasterizer/rasterizer.hpp"

namespace tinygs {

struct FastGSRasterizerParams {
  float f16_grad_scaler = 1.0f;
  bool enable_pose_opt = false;

  void from_json(const json& params);
  json to_json() const;
};

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

  FastGSRasterizerParams m_params;
};

}