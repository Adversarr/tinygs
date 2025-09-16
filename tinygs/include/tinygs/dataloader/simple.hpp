#pragma once
#include <tinygs/cuda/gpu_memory.hpp>
#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/random/pcg32.hpp"

namespace tinygs {

/// @brief Basic dataloader without acceleration
class SimpleDataLoader : public DataLoaderBase {
public:
  explicit SimpleDataLoader(std::shared_ptr<DatasetBase> dataset);

  ~SimpleDataLoader() = default;

  /// @brief Get next batch from dataset
  /// @param stream CUDA stream for data transfer
  GPUBatchInputOutput next(cudaStream_t stream) override;
  GPUBatchInputOutput next() override;

  /// Set the parameters for the dataloader.
  void set_params(const json &params) override;

  /// Get the parameters for the dataloader.
  json get_params() const override;

private:
  GPUMemory<float> m_gpu_memory;  ///< GPU buffer for data storage
  pcg32 m_rng;
};

}  // namespace tinygs
