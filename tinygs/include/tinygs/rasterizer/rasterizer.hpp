#pragma once
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/dataloader/dataloader.hpp"

namespace tinygs {

struct RasterizeParamsRuntime {
  /// If true, the rasterizer will prepare the gradients for 
  ///    - mat3x3 K;
  ///    - mat4x4 w2c;
  /// provided in GPUBatchInput.
  bool prepare_input_gradients = false;

  /// If true, the rasterizer will not store/compute the extra information
  /// required for backpropagation.
  bool inference = false;

  /// Put all the computation to this stream
  cudaStream_t stream = nullptr;

  GPUBatchInput fwd_input;
  GPUBatchOutput fwd_output;
  std::shared_ptr<GPUGaussian3d> gaussians_grad;
};

class RasterizerBase {
public:
  RasterizerBase();

  virtual ~RasterizerBase() = default;

  virtual void forward(const RasterizeParamsRuntime& params) = 0;
  virtual void backward(const RasterizeParamsRuntime& params) = 0;

  /// Called when the gaussians are changed. (especially the number of gaussians)
  virtual void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians);

  std::shared_ptr<GPUMemoryArena> memory_arena() { return m_memory_arena; }

protected:
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  std::shared_ptr<GPUMemoryArena> m_memory_arena;
};

}
