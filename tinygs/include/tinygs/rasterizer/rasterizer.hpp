#pragma once
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/core/gaussian.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/dataloader/dataloader.hpp"

namespace tinygs {

struct RasterizeContext {
  /// @brief Prepare gradients for camera intrinsics and extrinsics
  bool prepare_input_gradients = false;

  /// @brief Skip backpropagation information storage
  bool inference = false;

  /// @brief CUDA stream for computation
  cudaStream_t stream = nullptr;

  float grad_scaler = 1.0f;

  GPUBatchInput fwd_input;
  GPUBatchOutput fwd_output;
  GPUBatchInput grad_input;
  GPUBatchOutput grad_output;
  std::shared_ptr<GPUGaussian3d> gaussians_grad;

  /// @brief Densification information storage
  mutable std::shared_ptr<GPUBuffer<DensificationInfo>> densification_info;
};

class RasterizerBase {
public:
  RasterizerBase();

  virtual ~RasterizerBase() = default;

  virtual void forward(const RasterizeContext& params) = 0;

  virtual void backward(RasterizeContext& params) = 0;

  /// @brief Update gaussians when changed
  virtual void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians);

  virtual json get_params() const = 0;
  virtual void set_params(const json& j) = 0;

protected:
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUMemoryArena> m_memory_arena;
};

/// @brief Factory function for creating rasterizers
std::unique_ptr<RasterizerBase> create_rasterizer(const std::string& rasterizer_type);

}
