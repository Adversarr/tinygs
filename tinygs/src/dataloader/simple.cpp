#include "tinygs/dataloader/simple.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include <nvtx3/nvtx3.hpp>
#include "tinygs/dataloader/nvtx_dl.h"

namespace tinygs {

SimpleDataLoader::SimpleDataLoader(BackendRuntime& runtime, std::shared_ptr<DatasetBase> dataset) 
  : DataLoaderBase(runtime, dataset), m_current_index(0) {
  m_rng.seed(0);
  generate_permutation();
  QueueDesc queue_desc;
  queue_desc.non_blocking = true;
  queue_desc.debug_name = "SimpleDataLoader::transfer_queue";
  const auto queue_result = m_runtime->create_queue(queue_desc);
  if (!queue_result.ok()) {
    throw std::runtime_error(
        "SimpleDataLoader: failed to create transfer queue: " + to_string(queue_result.error()));
  }
  m_transfer_queue = queue_result.value();
  // Preallocate maximum GPU buffer once to avoid future reallocations
  size_t max_stride = m_dataset->image_shape().padded_size();
  m_gpu_memory = create_device_buffer(*m_runtime, max_stride * sizeof(float), "SimpleDataLoader::m_gpu_memory");
}

void SimpleDataLoader::generate_permutation() {
  size_t dataset_size = m_dataset->size();
  m_permutation.resize(dataset_size);
  
  // Initialize permutation with sequential indices
  for (size_t i = 0; i < dataset_size; ++i) {
    m_permutation[i] = i;
  }
  
#ifdef NDEBUG
  // Fisher-Yates shuffle using our RNG
  for (size_t i = dataset_size - 1; i > 0; --i) {
    size_t j = m_rng.next_uint(i + 1);
    std::swap(m_permutation[i], m_permutation[j]);
  }
#endif
  
  // Reset current index to start of new permutation
  m_current_index = 0;
}

GPUBatchInputOutput SimpleDataLoader::next(const BackendQueue* queue) {
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
  gpu_input.height = m_output_shape.height;
  gpu_input.width = m_output_shape.width;
  gpu_input.near = 0.1f; // Default near plane
  gpu_input.far = 100.0f; // Default far plane
  gpu_input.K = host_data.K;
  gpu_input.w2c = host_data.w2c;
  gpu_input.timestamp = host_data.timestamp;
  // GPU memory is preallocated to maximum stride at initialization; no resize here
  
  // Create GPU image structure
  Image gpu_image;
  gpu_image.shape = m_output_shape;
  gpu_image.data_type = m_params.data_type;
  gpu_image.data = m_gpu_memory->data();

  // Transfer data from host to GPU using the provided queue
  transfer_gpu(queue, gpu_image, host_data.image);

  // Prepare GPU batch output
  GPUBatchOutput gpu_output;
  gpu_output.image = Image{gpu_image.shape, gpu_image.data_type, gpu_image.data};

  return GPUBatchInputOutput{gpu_input, gpu_output};
}

GPUBatchInputOutput SimpleDataLoader::next() {
  DL_FUNC_RANGE();

  CHECK_THROW(m_transfer_queue != nullptr);
  auto r = next(m_transfer_queue.get());
  auto status = m_runtime->synchronize_queue(*m_transfer_queue);
  if (!status.ok()) {
    throw std::runtime_error("SimpleDataLoader::next() queue sync failed: " + to_string(status));
  }
  return std::move(r);
}

void SimpleDataLoader::set_params(const json &params) {
  if (params.contains("seed")) {
    m_rng.seed(params["seed"].get<uint64_t>());
  }
  DataLoaderBase::set_params(params);
}

json SimpleDataLoader::get_params() const {
  json params = DataLoaderBase::get_params();
  params["type"] = "simple";
  return params;
}

void SimpleDataLoader::reset() {
  // Ensure base preallocations (scratch buffer) happen once
  DataLoaderBase::reset();
  generate_permutation();
}

} // namespace tinygs
