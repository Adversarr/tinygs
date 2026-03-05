#pragma once
#include "tinygs/core/image.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/dataset/dataset.hpp"
#include "tinygs/platform/backend_types.hpp"

namespace tinygs {

/// @brief Camera parameters and metadata for a single training sample, residing on host.
///        Copied into RasterizeContext::fwd_input before each forward pass.
struct GPUBatchInput {
  uint32_t width, height; ///< Render resolution (may differ from dataset if progressive)
  float near, far;        ///< Near/far clipping planes
  mat3x3 K;               ///< Camera intrinsic matrix
  mat4x4 w2c;             ///< World-to-camera extrinsic matrix
  uuid_t timestamp;       ///< Unique sample identifier (used for pose optimization keys)
};

/// @brief Output images for a single sample (GPU memory).
struct GPUBatchOutput {
  Image image; ///< RGB image in GPU memory (Float32 or Float16 CHW-tiled layout)
};

/// @brief Combined input/output for one training sample.
struct GPUBatchInputOutput {
  GPUBatchInput input;
  GPUBatchOutput output;
};

/// @brief Serializable parameters shared by all dataloader implementations.
struct DataLoaderParams {
  DataType data_type = DataType::Float32; ///< Precision of the GPU output image

  void from_json(const json& params);
  json to_json() const;
};

/// @brief Abstract base class for dataloaders that feed training samples to the Orchestrator.
///
/// A dataloader wraps a DatasetBase and handles:
///   - Host→GPU transfer of images (with optional UInt8→Float conversion).
///   - Optional resolution down-sampling via `set_output_shape()`.
///   - Iteration order (sequential, shuffled, async pre-fetching, etc.).
///
/// Implementations: "simple" (synchronous), "async" (double-buffered pre-fetch).
class DataLoaderBase {
public:
  explicit DataLoaderBase(std::shared_ptr<DatasetBase> dataset);

  virtual ~DataLoaderBase() = default;

  /// @brief Return the next sample with its image already in GPU memory.
  ///        Cycles back to the beginning when the dataset is exhausted.
  virtual GPUBatchInputOutput next() = 0;

  /// @brief Restart iteration from the first sample.
  virtual void reset();

  /// @brief Set the desired output image resolution.
  /// @param shape Desired {width, height, channel}.  Only width/height are used;
  ///              the image will be rescaled on the host side before transfer.
  virtual void set_output_shape(const ImageShape& shape);

  /**
   * @brief Transfers image data from host memory to GPU memory with optional type conversion.
   *
   * This function performs asynchronous memory transfer from host to GPU with the following features:
   * - Validates that both GPU and host memory are allocated
   * - Enforces CHW (Channel-Height-Width) image format for both source and destination
   * - Handles automatic type conversion from UInt8 to float when data types differ
   * - Uses optimized vectorized conversion (packed4) when total elements are divisible by 4
   * - Requires matching image shapes between source and destination (throws on mismatch)
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
  void transfer_gpu(BackendStream stream, const Image& gpu_data, const Image& host_data);

  void transfer_gpu(const Image &gpu_data, const Image &host_data);

  /// Get the underlying dataset. It ensures the dataset is not null.
  std::shared_ptr<DatasetBase> get_dataset() const;

  /// Set the parameters for the dataloader.
  virtual void set_params(const json &params);

  /// Get the parameters for the dataloader.
  virtual json get_params() const;

protected:
  std::shared_ptr<DatasetBase> m_dataset;
  ImageShape m_output_shape;
  DataLoaderParams m_params;

private:
  GPUMemory<char> m_raw_data;
  std::mutex m_mutex;
};


/// @brief Create a dataloader object
/// @param dataloader_type Type of dataloader ("simple", etc.)
/// @param dataset Dataset to use with the dataloader
std::unique_ptr<DataLoaderBase> create_dataloader(const std::string& dataloader_type,
                                                  std::shared_ptr<DatasetBase> dataset);

}  // namespace tinygs
