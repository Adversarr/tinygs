#pragma once
#include "tinygs/dataset/dataset.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/core/image.hpp"



namespace tinygs {

struct GPUBatchInput {
  uint32_t batch_size; // Only 1 is support for now.
  uint32_t height, width;
  float near, far;
  mat3x3 K;
  mat4x4 w2c;
  // // Ks is nullptr, K is used. Use Ks otherwise.
  // const float* Ks;
};

struct GPUBatchOutput {
  Image<const float> image;
};

/**
 * @brief Forward & Backward Ready data for 3DGS.
 *
 */
struct GPUBatchInputOutput {
  GPUBatchInput input;
  GPUBatchOutput output;
};

class DataLoaderBase {
public:
  explicit DataLoaderBase(std::shared_ptr<DatasetBase> dataset) : m_dataset(dataset) {}

  virtual ~DataLoaderBase() = default;

  /**
   * @brief Get next batch of data ready for compute.
   * 
   * @param stream CUDA stream to use for asynchronous data transfers
   * @return GPUBatchInputOutput 
   */
  virtual GPUBatchInputOutput next(cudaStream_t stream) noexcept = 0;

  /**
   * @brief Get next batch of data ready for compute using default stream.
   * 
   * @return GPUBatchInputOutput 
   */
  virtual GPUBatchInputOutput next() noexcept {
    return next(cudaStreamDefault);
  }

  virtual void reset() {};

protected:
  std::shared_ptr<DatasetBase> m_dataset;
  size_t m_batch_size = 1; /// TODO: must be 1.
};

/**
 * @brief Helper function to transfer data from host to GPU.
 * 
 * @param stream CUDA stream to use for asynchronous transfer.
 * @param gpu_data GPU batch input/output data structure.
 * @param host_data Host batch input/output data structure.
 */
void transfer_gpu(cudaStream_t stream, const Image<float>& gpu_data, const Image<const float>& host_data);

inline void transfer_gpu(const Image<float>& gpu_data, const Image<const float>& host_data) {
  transfer_gpu(cudaStreamDefault, gpu_data, host_data);
}

} // namespace tinygs
