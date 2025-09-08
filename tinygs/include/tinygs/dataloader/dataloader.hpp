#pragma once
#include "tinygs/core/image.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

struct GPUBatchInput {
  uint32_t width, height;
  float near, far;
  mat3x3 K;
  mat4x4 w2c;
};

struct GPUBatchOutput {
  Image image;
  Image alpha;
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
  virtual GPUBatchInputOutput next(cudaStream_t stream) = 0;

  /**
   * @brief Get next batch of data ready for compute using default stream.
   *
   * @return GPUBatchInputOutput
   */
  virtual GPUBatchInputOutput next() = 0;

  virtual void reset() {};

  /**
  * @brief Helper function to transfer data from host to GPU.
  *
  * @param stream CUDA stream to use for asynchronous transfer.
  * @param gpu_data GPU batch input/output data structure.
  * @param host_data Host batch input/output data structure.
  */
  void transfer_gpu(cudaStream_t stream, const Image& gpu_data, const Image& host_data);

  void transfer_gpu(const Image &gpu_data, const Image &host_data);

protected:
  std::shared_ptr<DatasetBase> m_dataset;

private:
  GPUMemory<char> m_raw_data;
};


}  // namespace tinygs
