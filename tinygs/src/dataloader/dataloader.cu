#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include "tinygs/dataloader/simple.hpp"
#include "tinygs/dataloader/async.hpp"
#include "tinygs/common.hpp"
#include <algorithm>
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

constexpr float CHAR_TO_FLOAT = 1.0f / 255.0f;

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

// Nearest-neighbor resize from tiled CHW UInt8 to tiled CHW Float32
__global__ static void resize_nn_u8_to_float_tiled(
    const uint8_t* __restrict__ src, float* __restrict__ dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h,
    uint32_t src_tiled_w, uint32_t dst_tiled_w,
    uint32_t src_channel_stride, uint32_t dst_channel_stride,
    uint32_t channels) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;
  const uint32_t src_x = min((uint32_t)((float)x * (float)src_w / (float)dst_w), src_w - 1);
  const uint32_t src_y = min((uint32_t)((float)y * (float)src_h / (float)dst_h), src_h - 1);
  const uint32_t dst_idx_base = get_linear_index_tiled(y, x, dst_tiled_w);
  const uint32_t src_idx_base = get_linear_index_tiled(src_y, src_x, src_tiled_w);
  constexpr float CHAR_TO_FLOAT = 1.0f / 255.0f;
  for (uint32_t c = 0; c < channels; ++c) {
    const uint8_t v = src[src_idx_base + c * src_channel_stride];
    dst[dst_idx_base + c * dst_channel_stride] = CHAR_TO_FLOAT * (float)v;
  }
}

// Nearest-neighbor resize from tiled CHW Float32 to tiled CHW Float32
__global__ static void resize_nn_float_to_float_tiled(
    const float* __restrict__ src, float* __restrict__ dst,
    uint32_t src_w, uint32_t src_h,
    uint32_t dst_w, uint32_t dst_h,
    uint32_t src_tiled_w, uint32_t dst_tiled_w,
    uint32_t src_channel_stride, uint32_t dst_channel_stride,
    uint32_t channels) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) return;
  const uint32_t src_x = min((uint32_t)((float)x * (float)src_w / (float)dst_w), src_w - 1);
  const uint32_t src_y = min((uint32_t)((float)y * (float)src_h / (float)dst_h), src_h - 1);
  const uint32_t dst_idx_base = get_linear_index_tiled(y, x, dst_tiled_w);
  const uint32_t src_idx_base = get_linear_index_tiled(src_y, src_x, src_tiled_w);
  for (uint32_t c = 0; c < channels; ++c) {
    const float v = src[src_idx_base + c * src_channel_stride];
    dst[dst_idx_base + c * dst_channel_stride] = v;
  }
}

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

void DataLoaderBase::transfer_gpu(cudaStream_t stream, const Image &gpu_data,
                                  const Image &host_data) {
  std::lock_guard lock(m_mutex);
  // TODO: support multiple resolution for dataloader
  NVTX3_FUNC_RANGE();
  if (gpu_data.data == nullptr) {
    throw std::runtime_error("GPU memory is not allocated.");
  } else if (host_data.data == nullptr) {
    throw std::runtime_error("Host memory is not allocated.");
  }

  const uint32_t src_width = host_data.shape.width;
  const uint32_t src_height = host_data.shape.height;
  const uint32_t dst_width = gpu_data.shape.width;
  const uint32_t dst_height = gpu_data.shape.height;
  const uint32_t channels = gpu_data.shape.channel;
  const bool same_shape = (src_width == dst_width) && (src_height == dst_height) && (host_data.shape.channel == gpu_data.shape.channel);

  if (gpu_data.shape != m_output_shape) {
    throw std::runtime_error(fmt::format(
      "Image shape mismatch not supported: gpu_data.shape={}, m_output_shape={}",
      to_string(gpu_data.shape), to_string(m_output_shape)));
  }
  // Helper for per-channel stride in tiled CHW layout
  auto compute_channel_stride = [](uint32_t w, uint32_t h) {
    const uint32_t w_in_tile = (w + kImageTileMask) >> kImageTileLog2;
    const uint32_t h_in_tile = (h + kImageTileMask) >> kImageTileLog2;
    return w_in_tile * h_in_tile << (2 * kImageTileLog2);
  };

  if (host_data.data_type == gpu_data.data_type && same_shape) {
    // Same type & shape: direct copy
    const size_t bytes = static_cast<size_t>(gpu_data.shape.padded_size()) * sizeof(float);
    CUDA_CHECK_THROW(cudaMemcpyAsync(gpu_data.data, host_data.data, bytes,
                                     cudaMemcpyHostToDevice, stream));
  } else if (host_data.data_type == ImageDataType::UInt8) {
    // Copy source bytes to device scratch
    const size_t src_total_bytes = static_cast<size_t>(host_data.shape.padded_size()) * sizeof(uint8_t);
    if (m_raw_data.size() < src_total_bytes) {
      throw std::runtime_error("Internal GPU scratch buffer insufficient; preallocate via reset().");
    }
    char* raw_data = m_raw_data.data();
    CUDA_CHECK_THROW(cudaMemcpyAsync(raw_data, host_data.data, src_total_bytes,
                                     cudaMemcpyHostToDevice, stream));

    if (same_shape) {
      // Convert only
      const uint32_t total_elements = host_data.shape.padded_size();
      if (total_elements % 4 == 0){
        convert_u8_float_packed4<<<(total_elements / 4 + 255) / 256, 256, 0, stream>>>(
          reinterpret_cast<const unsigned char*>(raw_data), reinterpret_cast<float *>(gpu_data.data), total_elements / 4);
      } else {
        convert_u8_float<<<(total_elements + 255) / 256, 256, 0, stream>>>(
          reinterpret_cast<const unsigned char*>(raw_data), reinterpret_cast<float *>(gpu_data.data), total_elements);
      }
    } else {
      // Resize + convert (nearest) in tiled CHW
      if (host_data.shape.channel != channels) {
        throw std::runtime_error("Channel mismatch (host vs gpu) not supported for resize.");
      }
      const uint32_t dst_tiled_w = (dst_width + kImageTileMask) >> kImageTileLog2;
      const uint32_t src_tiled_w = (src_width + kImageTileMask) >> kImageTileLog2;
      const uint32_t dst_channel_stride = compute_channel_stride(dst_width, dst_height);
      const uint32_t src_channel_stride = compute_channel_stride(src_width, src_height);
      dim3 block(16, 16);
      dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
      resize_nn_u8_to_float_tiled<<<grid, block, 0, stream>>>(reinterpret_cast<const uint8_t*>(raw_data),
                                                              reinterpret_cast<float*>(gpu_data.data),
                                                              src_width, src_height, dst_width, dst_height,
                                                              src_tiled_w, dst_tiled_w,
                                                              src_channel_stride, dst_channel_stride,
                                                              channels);
    }
  } else if (host_data.data_type == ImageDataType::Float32) {
    if (same_shape) {
      const size_t bytes = static_cast<size_t>(gpu_data.shape.padded_size()) * sizeof(float);
      CUDA_CHECK_THROW(cudaMemcpyAsync(gpu_data.data, host_data.data, bytes,
                                       cudaMemcpyHostToDevice, stream));
    } else {
      // Copy source floats to device scratch and resize (nearest)
      const size_t src_total_bytes = static_cast<size_t>(host_data.shape.padded_size()) * sizeof(float);
      if (m_raw_data.size() < src_total_bytes) {
        throw std::runtime_error("Internal GPU scratch buffer insufficient; preallocate via reset().");
      }
      char* raw_data = m_raw_data.data();
      CUDA_CHECK_THROW(cudaMemcpyAsync(raw_data, host_data.data, src_total_bytes,
                                       cudaMemcpyHostToDevice, stream));
      if (host_data.shape.channel != channels) {
        throw std::runtime_error("Channel mismatch (host vs gpu) not supported for resize.");
      }
      const uint32_t dst_tiled_w = (dst_width + kImageTileMask) >> kImageTileLog2;
      const uint32_t src_tiled_w = (src_width + kImageTileMask) >> kImageTileLog2;
      const uint32_t dst_channel_stride = compute_channel_stride(dst_width, dst_height);
      const uint32_t src_channel_stride = compute_channel_stride(src_width, src_height);
      dim3 block(16, 16);
      dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
      resize_nn_float_to_float_tiled<<<grid, block, 0, stream>>>(reinterpret_cast<const float*>(raw_data),
                                                                 reinterpret_cast<float*>(gpu_data.data),
                                                                 src_width, src_height, dst_width, dst_height,
                                                                 src_tiled_w, dst_tiled_w,
                                                                 src_channel_stride, dst_channel_stride,
                                                                 channels);
    }
  } else {
    throw std::runtime_error("Unsupported host image data type for transfer.");
  }
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void DataLoaderBase::transfer_gpu(const Image &gpu_data,
                                  const Image &host_data) {
  transfer_gpu(nullptr, gpu_data, host_data);
}

std::shared_ptr<DatasetBase> DataLoaderBase::get_dataset() const {
  if (! m_dataset) {
    throw std::runtime_error("Dataset not set.");
  }
  return m_dataset;
}

void DataLoaderBase::reset() {
  NVTX3_FUNC_RANGE();
  // Preallocate device scratch buffer to the maximum dataset image size in bytes
  if (m_dataset) {
    const size_t max_elements = static_cast<size_t>(m_dataset->image_shape().padded_size());
    const size_t max_bytes = max_elements * sizeof(float); // reserve enough for float-sized scratch
    if (m_raw_data.size() < max_bytes) {
      m_raw_data.resize(max_bytes);
    }
  }
}

std::unique_ptr<DataLoaderBase> create_dataloader(const std::string& dataloader_type,
                                                  std::shared_ptr<DatasetBase> dataset) {
  std::string lower_dataloader_type = to_lower(dataloader_type);

  if (lower_dataloader_type == "simple") {
    return std::make_unique<SimpleDataLoader>(dataset);
  } else if (lower_dataloader_type == "async") {
    return std::make_unique<AsyncDataLoader>(dataset);
  } else {
    throw std::runtime_error("Unknown dataloader type: " + dataloader_type);
  }
}

void DataLoaderBase::set_params(const json &params) { (void)params; }

json DataLoaderBase::get_params() const { return json::object(); }

DataLoaderBase::DataLoaderBase(std::shared_ptr<DatasetBase> dataset) : m_dataset(dataset) {
  m_output_shape = dataset->image_shape();
}

}  // namespace tinygs