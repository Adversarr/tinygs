#include "tinygs/dataloader/simple.hpp"
#include "tinygs/cuda/common_host.hpp"
#include <random>
namespace tinygs {

SimpleDataLoader::SimpleDataLoader(std::shared_ptr<DatasetBase> dataset) : DataLoaderBase(dataset), m_current_index(0) {
  m_rng.seed(0);
  generate_permutation();
}

void SimpleDataLoader::generate_permutation() {
  size_t dataset_size = m_dataset->size();
  m_permutation.resize(dataset_size);
  
  // Initialize permutation with sequential indices
  for (size_t i = 0; i < dataset_size; ++i) {
    m_permutation[i] = i;
  }
  
  // Fisher-Yates shuffle using our RNG
  for (size_t i = dataset_size - 1; i > 0; --i) {
    size_t j = m_rng.next_uint(i + 1);
    std::swap(m_permutation[i], m_permutation[j]);
  }
  
  // Reset current index to start of new permutation
  m_current_index = 0;
}

GPUBatchInputOutput SimpleDataLoader::next(cudaStream_t stream) {
  // Check if we've consumed the entire permutation
  if (m_current_index >= m_permutation.size()) {
    // Generate a new permutation and reset index
    generate_permutation();
  }
  
  // Get the next index from the current permutation
  size_t current_index = m_permutation[m_current_index];
  m_current_index++;
  
  Data host_data = (*m_dataset)[current_index];

  // Prepare GPU batch input
  GPUBatchInput gpu_input;
  gpu_input.height = host_data.image.shape.height;
  gpu_input.width = host_data.image.shape.width;
  gpu_input.near = 0.1f; // Default near plane
  gpu_input.far = 100.0f; // Default far plane
  gpu_input.K = host_data.K;
  gpu_input.w2c = host_data.w2c;
  gpu_input.timestamp = host_data.timestamp;
  // Allocate GPU memory for the image if needed
  size_t image_size = gpu_input.height * gpu_input.width * host_data.image.shape.channel;
  m_gpu_memory.resize(image_size);
  
  // Create GPU image structure
  Image gpu_image;
  gpu_image.shape = host_data.image.shape; // TODO: allow lower resolution.
  gpu_image.data_type = ImageDataType::Float32;
  gpu_image.data = m_gpu_memory.data();

  // Transfer data from host to GPU using the provided CUDA stream
  transfer_gpu(stream, gpu_image, host_data.image);

  // Prepare GPU batch output
  GPUBatchOutput gpu_output;
  gpu_output.image = Image{gpu_image.shape, gpu_image.data_type, gpu_image.data};

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

void SimpleDataLoader::reset() {
  generate_permutation();
}

} // namespace tinygs