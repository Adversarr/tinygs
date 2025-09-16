#include "tinygs/dataloader/simple.hpp"
#include "tinygs/cuda/common_host.hpp"
#include <random>
namespace tinygs {

SimpleDataLoader::SimpleDataLoader(std::shared_ptr<DatasetBase> dataset) : DataLoaderBase(dataset) {
  m_rng.seed(0);
}

GPUBatchInputOutput SimpleDataLoader::next(cudaStream_t stream) {
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
  Image gpu_image;
  gpu_image.shape = host_data.image.shape;
  gpu_image.format = host_data.image.format;
  gpu_image.data_type = ImageDataType::Float32;
  gpu_image.data = m_gpu_memory.data();

  // Transfer data from host to GPU using the provided CUDA stream
  transfer_gpu(stream, gpu_image, host_data.image);

  // Prepare GPU batch output
  GPUBatchOutput gpu_output;
  gpu_output.image = Image{
    gpu_image.shape,
    gpu_image.format,
    gpu_image.data_type,
    gpu_image.data
  };

  return GPUBatchInputOutput{gpu_input, gpu_output};
}

GPUBatchInputOutput SimpleDataLoader::next() {
  auto r = next(cudaStreamDefault);
  CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamDefault));
  return std::move(r);
}

void SimpleDataLoader::set_params(const json &params) {
  if (params.contains("seed")) {
    m_rng.seed(params["seed"].get<uint64_t>());
  }
}

json SimpleDataLoader::get_params() const {
  json params = json::object();
  params["type"] = "simple";
  return params;
}

} // namespace tinygs