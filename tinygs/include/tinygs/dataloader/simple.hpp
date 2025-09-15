#pragma once
#include <tinygs/cuda/gpu_memory.hpp>
#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/random/pcg32.hpp"

namespace tinygs {

// The most baseline dataloader, no any acceleration.
class SimpleDataLoader : public DataLoaderBase {
public:
  explicit SimpleDataLoader(std::shared_ptr<DatasetBase> dataset);

  ~SimpleDataLoader() = default;

  /**
   * @brief Get the next batch of data from the dataset.
   * 
   * @param stream The CUDA stream to use for the data transfer.
   * @return GPUBatchInputOutput The next batch of data.
   */
  GPUBatchInputOutput next(cudaStream_t stream) override;
  GPUBatchInputOutput next() override;

  /// Set the parameters for the dataloader.
  void set_params(const json &params) override;

  /// Get the parameters for the dataloader.
  json get_params() const override;

private:
  // It always use this buffer to store the data on GPU.
  GPUMemory<float> m_gpu_memory;
  pcg32 m_rng;
};

}  // namespace tinygs
