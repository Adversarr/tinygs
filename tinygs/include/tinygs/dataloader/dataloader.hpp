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

  virtual void reset();

  /**
   * @brief Transfers image data from host memory to GPU memory with optional type conversion.
   *
   * This function performs asynchronous memory transfer from host to GPU with the following features:
   * - Validates that both GPU and host memory are allocated
   * - Enforces CHW (Channel-Height-Width) image format for both source and destination
   * - Handles automatic type conversion from UInt8 to float when data types differ
   * - Uses optimized vectorized conversion (packed4) when total elements are divisible by 4
   * - Requires matching image shapes between source and destination
   *
   * @param stream CUDA stream to use for asynchronous transfer operations
   * @param gpu_data Destination GPU image data structure (must be pre-allocated)
   * @param host_data Source host image data structure containing the data to transfer
   *
   * @throws std::runtime_error if GPU or host memory is not allocated
   * @throws std::runtime_error if image format is not CHW for either source or destination
   * @throws std::runtime_error if image shapes don't match between source and destination
   *
   * @note When data types differ, assumes host data is UInt8 and converts to float on GPU
   * @note Uses internal buffer (m_raw_data) for intermediate storage during type conversion
   */
  void transfer_gpu(cudaStream_t stream, const Image& gpu_data, const Image& host_data);

  void transfer_gpu(const Image &gpu_data, const Image &host_data);

protected:
  std::shared_ptr<DatasetBase> m_dataset;

private:
  GPUMemory<char> m_raw_data;
};


}  // namespace tinygs
