#include "tinygs/dataloader/simple.hpp"
#include "tinygs/cuda/common_host.hpp"
#include <random>
namespace tinygs {

SimpleDataLoader::SimpleDataLoader(std::shared_ptr<DatasetBase> dataset) : DataLoaderBase(dataset) {
  m_rng.seed(0); // TODO: make it configurable
}

GPUBatchInputOutput SimpleDataLoader::next(cudaStream_t stream) noexcept {
#ifdef NDEBUG
  // Randomly pick a data from the dataset
  size_t current_index = m_rng.next_uint(m_dataset->size());
#else
  // Get the next data from the dataset
  static size_t current_index = 0;
  current_index += 1;
  if (current_index >= m_dataset->size()) {
    current_index = 0; // Loop back to the beginning
  }
#endif
  Data host_data = (*m_dataset)[current_index];

  // Prepare GPU batch input
  GPUBatchInput gpu_input;
  gpu_input.batch_size = 1; // Only batch size 1 is supported
  gpu_input.height = host_data.image.shape.height;
  gpu_input.width = host_data.image.shape.width;
  gpu_input.near = 0.1f; // Default near plane
  gpu_input.far = 100.0f; // Default far plane
  gpu_input.K = host_data.K;
  gpu_input.w2c = host_data.w2c;
  
  // Allocate GPU memory for the image if needed
  size_t image_size = gpu_input.height * gpu_input.width * host_data.image.shape.channel;
  m_gpu_memory.resize(image_size);
  
  // Create GPU image structure
  Image<float> gpu_image;
  gpu_image.shape = host_data.image.shape;
  gpu_image.format = host_data.image.format;
  gpu_image.data = m_gpu_memory.data();
  
  // Transfer data from host to GPU using the provided CUDA stream
  // Create a non-const version of host image for transfer
  Image<const float> host_image_mutable;
  host_image_mutable.shape = host_data.image.shape;
  host_image_mutable.format = host_data.image.format;
  host_image_mutable.data = host_data.image.data;
  transfer_gpu(stream, gpu_image, host_image_mutable);
  
  // Prepare GPU batch output
  GPUBatchOutput gpu_output;
  gpu_output.image = Image<float>{
    gpu_image.shape,
    gpu_image.format,
    gpu_image.data
  };
  
  return GPUBatchInputOutput{gpu_input, gpu_output};
}

}