#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include "tinygs/dataloader/simple.hpp"
#include <algorithm>

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

void DataLoaderBase::transfer_gpu(cudaStream_t stream, const Image &gpu_data,
                                  const Image &host_data) {
  TINYGS_TIMER("DataLoaderBase::transfer_gpu");
  if (gpu_data.data == nullptr) {
    throw std::runtime_error("GPU memory is not allocated.");
  } else if (host_data.data == nullptr) {
    throw std::runtime_error("Host memory is not allocated.");
  }

  // For the DL based tasks, we only support CHW
  if (gpu_data.format != ImageFormat::CHW || host_data.format != ImageFormat::CHW) {
    throw std::runtime_error("Only CHW format is supported.");
  }

  uint32_t height = gpu_data.shape.height;
  uint32_t width = gpu_data.shape.width;
  uint32_t channels = gpu_data.shape.channel;
  uint32_t total_elements = height * width * channels;


  
  if (gpu_data.shape != host_data.shape) {
    throw std::runtime_error(fmt::format(
      "Image shape mismatch not supported: gpu_data.shape={}, "
      "host_data.shape={}",
      to_string(gpu_data.shape), to_string(host_data.shape)));
  }

  if (host_data.data_type == gpu_data.data_type) {
    // everything is good.
    CUDA_CHECK_THROW(cudaMemcpyAsync(gpu_data.data, host_data.data,
                                      total_elements * sizeof(float),
                                      cudaMemcpyHostToDevice, stream));
  } else {
    // not same type, we need to "unzip" the host data.
    assert(host_data.data_type == ImageDataType::UInt8);
    if (m_raw_data.size() < total_elements * sizeof(char)) {
      // This operation is not happening frequently, we block all the streams
      // and allocate the memory
      m_raw_data.resize(total_elements * sizeof(char));
    }

    char* raw_data = m_raw_data.data();
    CUDA_CHECK_THROW(cudaMemcpyAsync(raw_data, host_data.data,
                                     total_elements * sizeof(char),
                                     cudaMemcpyHostToDevice, stream));

    // now we convert the raw data to float
    if (total_elements % 4 == 0){
      convert_u8_float_packed4<<<(total_elements / 4 + 255) / 256, 256, 0, stream>>>( //
        (const unsigned char*)raw_data, (float *)gpu_data.data, total_elements / 4);
    } else {
      convert_u8_float<<<(total_elements + 255) / 256, 256, 0, stream>>>( //
        (unsigned char*)raw_data, (float *)gpu_data.data, total_elements);
    }
  }
}

void DataLoaderBase::transfer_gpu(const Image &gpu_data,
                                  const Image &host_data) {
  transfer_gpu(cudaStreamDefault, gpu_data, host_data);
  CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamDefault));
}

std::shared_ptr<DatasetBase> DataLoaderBase::get_dataset() const {
  if (! m_dataset) {
    throw std::runtime_error("Dataset not set.");
  }
  return m_dataset;
}

void DataLoaderBase::reset() {
  // do nothing
}

std::unique_ptr<DataLoaderBase> create_dataloader(const std::string& dataloader_type,
                                                  std::shared_ptr<DatasetBase> dataset) {
  std::string lower_dataloader_type = to_lower(dataloader_type);

  if (lower_dataloader_type == "simple") {
    return std::make_unique<SimpleDataLoader>(dataset);
  } else {
    throw std::runtime_error("Unknown dataloader type: " + dataloader_type);
  }
}

void DataLoaderBase::set_params(const json &params) { (void)params; }

json DataLoaderBase::get_params() const { return json::object(); }
} // namespace tinygs