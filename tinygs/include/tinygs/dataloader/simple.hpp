#pragma once
#include <tinygs/cuda/gpu_memory.hpp>
#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/random/pcg32.hpp"

namespace tinygs {

// The most baseline dataloader, no any acceleration.
class SimpleDataLoader : public DataLoaderBase {
public:
  explicit SimpleDataLoader(std::shared_ptr<DatasetBase> dataset) : DataLoaderBase(dataset) {}

  ~SimpleDataLoader() = default;

  GPUBatchInputOutput next(cudaStream_t stream) noexcept override;
  GPUBatchInputOutput next() noexcept override {
    return next(cudaStreamDefault);
  }

private:
  // It always use this buffer to store the data on GPU.
  GPUMemory<float> m_gpu_memory;
  pcg32 m_rng;
};

}  // namespace tinygs
