#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include "tinygs/dataloader/simple.hpp"
#include "tinygs/dataloader/async.hpp"
#include "tinygs/common.hpp"
#include "tinygs/dataloader/nvtx_dl.h"
#include <cuda_fp16.h>

namespace tinygs {

namespace {

/// @brief Create or grow a shared BackendBuffer to at least the given size.
///        If the existing buffer is large enough, it is reused.
inline void ensure_buffer_size(
    const std::shared_ptr<BackendRuntime>& runtime,
    std::shared_ptr<BackendBuffer>& buf,
    size_t required_bytes,
    const std::string& debug_name) {
  if (!buf || buf->size_bytes() < required_bytes) {
    buf = create_device_buffer(runtime, required_bytes, debug_name);
  }
}

}  // namespace

// Conversion constant from 8-bit integer to floating point
constexpr float CHAR_TO_FLOAT = 1.0f / 255.0f;

////////////////////////////// Conversion Kernels //////////////////////////////

/// @brief Convert uint8_t image data to float32 on GPU
/// @param input Input uint8_t image data
/// @param output Output float32 image data
/// @param total Total number of elements to convert
__global__ static void convert_u8_float(
  const unsigned char* input,
  float* output,
  size_t total
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = CHAR_TO_FLOAT * (float) input[idx];
}

/// @brief Convert uint8_t image data to float32 on GPU using packed4 operations
/// @param input Input uint8_t image data
/// @param output Output float32 image data
/// @param total Total number of elements to convert (divided by 4)
__global__ static void convert_u8_float_packed4(
    const unsigned char* input,
    float* output,
    size_t total
) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {// 4 elements per thread
    return;
  }

  const uchar4 *in = reinterpret_cast<const uchar4 *>(input) + idx;
  const float4 y = {CHAR_TO_FLOAT * in->x, CHAR_TO_FLOAT * in->y,
                    CHAR_TO_FLOAT * in->z, CHAR_TO_FLOAT * in->w};
  float4 *out = reinterpret_cast<float4 *>(output) + idx;
  *out = y;
}

////////////////////////////// FP16 Conversion Kernels //////////////////////////////

/// @brief Convert uint8_t image data to half-precision float (FP16) on GPU
/// @param input Input uint8_t image data
/// @param output Output FP16 image data
/// @param total Total number of elements to convert
__global__ static void convert_u8_half(
  const unsigned char* input,
  __half* output,
  size_t total
) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  output[idx] = __float2half(CHAR_TO_FLOAT * static_cast<float>(input[idx]));
}

/// @brief Convert uint8_t image data to half-precision float (FP16) using packed4 operations
/// @param input Input uint8_t image data
/// @param output Output FP16 image data
/// @param total Total number of elements to convert (divided by 4)
__global__ static void convert_u8_half_packed4(
    const unsigned char* input,
    __half* output,
    size_t total
) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return; // 4 elements per thread
  const uchar4* in = reinterpret_cast<const uchar4*>(input) + idx;
  __half* out = output + (idx * 4);
  out[0] = __float2half(CHAR_TO_FLOAT * static_cast<float>(in->x));
  out[1] = __float2half(CHAR_TO_FLOAT * static_cast<float>(in->y));
  out[2] = __float2half(CHAR_TO_FLOAT * static_cast<float>(in->z));
  out[3] = __float2half(CHAR_TO_FLOAT * static_cast<float>(in->w));
}

////////////////////////////// DataLoaderBase Implementation //////////////////////////////

void DataLoaderParams::from_json(const json& params) {
  if (params.contains("data_type")) {
    std::string data_type_str = params["data_type"];
    data_type = from_string<DataType>(data_type_str);
  }
}

json DataLoaderParams::to_json() const {
  json params;
  params["data_type"] = to_string(data_type);
  return params;
}


/// @brief Update the output shape for the dataloader
void DataLoaderBase::set_output_shape(const ImageShape &shape) {
  {
    // Update shape under lock, but do not hold the lock while resetting
    std::lock_guard lock(m_mutex);
    m_output_shape = shape;
  }
  CUDA_CHECK_THROW(cudaDeviceSynchronize());
  m_dataset->get_camera_loader().resize_sensor(shape.width, shape.height);
  // Reset outside the lock to avoid deadlock with async prefetch thread
  reset(); // must reset, async dataloader have an individual thread to load the data.
  log_info("Updating output shape to {}", to_string(m_output_shape));
}

/// @brief Transfer image data from host to GPU with format conversion
/// @param stream CUDA stream for async operations
/// @param gpu_data Destination GPU image
/// @param host_data Source host image
void DataLoaderBase::transfer_gpu(BackendStream stream, const Image &gpu_data,
                                  const Image &host_data) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  std::lock_guard lock(m_mutex);
  // Avoid NVTX ranges here because this function is called from
  // both main and background threads.
  if (gpu_data.data == nullptr) {
    throw std::runtime_error("GPU memory is not allocated.");
  } else if (host_data.data == nullptr) {
    throw std::runtime_error("Host memory is not allocated.");
  }

  const bool same_shape = host_data.shape == gpu_data.shape;

  if (gpu_data.shape != m_output_shape) {
    throw std::runtime_error(fmt::format(
      "Image shape mismatch not supported: gpu_data.shape={}, m_output_shape={}",
      to_string(gpu_data.shape), to_string(m_output_shape)));
  }

  // Case 1: Same type & shape - direct copy
  if (host_data.data_type == gpu_data.data_type && same_shape) {
    const size_t elem_size =
      gpu_data.data_type == DataType::Float32 ? sizeof(float) :
      (gpu_data.data_type == DataType::Float16 ? sizeof(__half) : sizeof(uint8_t));
    const size_t bytes = static_cast<size_t>(gpu_data.shape.padded_size()) * elem_size;
    CUDA_CHECK_THROW(cudaMemcpyAsync(gpu_data.data, host_data.data, bytes,
                                     cudaMemcpyHostToDevice, cuda_stream));
  } 
  // Case 2: Converting from UInt8 host data
  else if (host_data.data_type == DataType::UInt8) {
    // Copy source bytes to device scratch
    const size_t src_total_bytes =
        static_cast<size_t>(host_data.shape.padded_size()) * sizeof(uint8_t);
    if (!m_raw_data || m_raw_data->size_bytes() < src_total_bytes) {
      throw std::runtime_error("Internal GPU scratch buffer insufficient; preallocate via reset().");
    }
    char *raw_data = static_cast<char*>(m_raw_data->data());
    CUDA_CHECK_THROW(cudaMemcpyAsync(raw_data, host_data.data,
                                     src_total_bytes, cudaMemcpyHostToDevice,
                                     cuda_stream));

    // Case 2a: Same shape - just convert data type
    if (same_shape) {
      const uint32_t total_elements = host_data.shape.padded_size();
      if (gpu_data.data_type == DataType::Float32) {
        // Convert UInt8 to Float32
        if (total_elements % 4 == 0) {
          convert_u8_float_packed4<<<(total_elements / 4 + 255) / 256, 256, 0,
                                     cuda_stream>>>(
              reinterpret_cast<const unsigned char *>(raw_data),
              reinterpret_cast<float *>(gpu_data.data), total_elements / 4);
        } else {
          convert_u8_float<<<(total_elements + 255) / 256, 256, 0, cuda_stream>>>(
              reinterpret_cast<const unsigned char *>(raw_data),
              reinterpret_cast<float *>(gpu_data.data), total_elements);
        }
      } 
      else if (gpu_data.data_type == DataType::Float16) {
        // Convert UInt8 to FP16
        if (total_elements % 4 == 0) {
          convert_u8_half_packed4<<<(total_elements / 4 + 255) / 256, 256, 0, cuda_stream>>>(
              reinterpret_cast<const unsigned char *>(raw_data),
              reinterpret_cast<__half *>(gpu_data.data), total_elements / 4);
        } else {
          convert_u8_half<<<(total_elements + 255) / 256, 256, 0, cuda_stream>>>(
              reinterpret_cast<const unsigned char *>(raw_data),
              reinterpret_cast<__half *>(gpu_data.data), total_elements);
        }
      } 
      else if (gpu_data.data_type == DataType::UInt8) {
        // Same-shape UInt8 -> UInt8: device-to-device copy
        CUDA_CHECK_THROW(cudaMemcpyAsync(gpu_data.data, raw_data,
                                         src_total_bytes, cudaMemcpyDeviceToDevice, cuda_stream));
      } else {
        throw std::runtime_error("Unsupported GPU image data type for UInt8 host.");
      }
    }
    // Case 2b: Different shape - not supported
    else {
      throw std::runtime_error(fmt::format(
        "Image shape mismatch not supported: host_data.shape={}, gpu_data.shape={}. "
        "Shapes must match exactly.",
        to_string(host_data.shape), to_string(gpu_data.shape)));
    }
  } 
  // Case 3: Host-side float32 data (not supported)
  else if (host_data.data_type == DataType::Float32) {
    throw std::runtime_error("Host-side float32 data type not supported for transfer.");
  } 
  // Case 4: Unsupported host data type
  else {
    throw std::runtime_error("Unsupported host image data type for transfer.");
  }
  CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));
}

/// @brief Transfer image data from host to GPU using default stream
/// @param gpu_data Destination GPU image
/// @param host_data Source host image
void DataLoaderBase::transfer_gpu(const Image &gpu_data,
                                  const Image &host_data) {
  transfer_gpu(nullptr, gpu_data, host_data);
}

/// @brief Get the dataset associated with this dataloader
std::shared_ptr<DatasetBase> DataLoaderBase::get_dataset() const {
  if (! m_dataset) {
    throw std::runtime_error("Dataset not set.");
  }
  return m_dataset;
}

/// @brief Reset the dataloader and preallocate GPU buffers
void DataLoaderBase::reset() {
  DL_FUNC_RANGE();
  // Preallocate device scratch buffer to the maximum dataset image size in bytes
  if (m_dataset) {
    const size_t max_elements = static_cast<size_t>(m_dataset->image_shape().padded_size());
    const size_t max_bytes = max_elements * sizeof(float); // reserve enough for float-sized scratch
    if (!m_raw_data || m_raw_data->size_bytes() < max_bytes) {
      ensure_buffer_size(m_runtime, m_raw_data, max_bytes, "DataLoaderBase::m_raw_data");
    }
  }
}

/// @brief Create a dataloader of the specified type
/// @param dataloader_type Type of dataloader to create ("simple" or "async")
/// @param runtime Backend runtime for GPU operations
/// @param dataset Dataset to load from
/// @return Unique pointer to the created dataloader
std::unique_ptr<DataLoaderBase> create_dataloader(const std::string& dataloader_type,
                                                   std::shared_ptr<BackendRuntime> runtime,
                                                   std::shared_ptr<DatasetBase> dataset) {
  std::string lower_dataloader_type = to_lower(dataloader_type);

  if (lower_dataloader_type == "simple") {
    return std::make_unique<SimpleDataLoader>(runtime, dataset);
  } else if (lower_dataloader_type == "async") {
    return std::make_unique<AsyncDataLoader>(runtime, dataset);
  }
  throw std::runtime_error("Unknown dataloader type: " + dataloader_type);
}

/// @brief Set parameters for the dataloader
void DataLoaderBase::set_params(const json &params) {
  DataLoaderParams dl_params;
  dl_params.from_json(params);
  m_params = dl_params;
}

/// @brief Get parameters from the dataloader
json DataLoaderBase::get_params() const {
  json params = m_params.to_json();
  return params;
}

/// @brief Constructor for DataLoaderBase
DataLoaderBase::DataLoaderBase(std::shared_ptr<BackendRuntime> runtime, std::shared_ptr<DatasetBase> dataset) 
  : m_runtime(runtime), m_dataset(dataset) {
  m_output_shape = dataset->image_shape();
  // m_raw_data starts null; allocated lazily in set_output_shape()
}

}  // namespace tinygs
