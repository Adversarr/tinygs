#include "tinygs/dataloader/dataloader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/cuda/common_device.cuh"

namespace tinygs {

void transfer_gpu(cudaStream_t stream, const Image<float> &gpu_data,
                  const Image<const float> &host_data) {
  if (gpu_data.data == nullptr) {
    throw std::runtime_error("GPU memory is not allocated.");
  } else if (host_data.data == nullptr) {
    throw std::runtime_error("Host memory is not allocated.");
  }

  uint32_t height = gpu_data.shape.height;
  uint32_t width = gpu_data.shape.width;
  uint32_t channels = gpu_data.shape.channels;
  uint32_t total_elements = height * width * channels;

  // TODO: support for multiple shapes
  if (gpu_data.shape != host_data.shape || gpu_data.format != host_data.format) {
    throw std::runtime_error(fmt::format(
        "Image shape or format mismatch not supported: gpu_data.shape={}, "
        "host_data.shape={}, gpu_data.format={}, host_data.format={}",
        to_string(gpu_data.shape), to_string(host_data.shape),
        to_string(gpu_data.format), to_string(host_data.format)));
  }
  
  // NOTE: assuming same shape and channels.
  cudaMemcpyAsync(gpu_data.data, host_data.data, total_elements * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
}

}  // namespace tinygs