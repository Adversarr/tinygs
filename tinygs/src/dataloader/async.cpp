#include "tinygs/dataloader/async.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/random/pcg32.hpp"
#include <queue>
#include <nvtx3/nvtx3.hpp>
#include "tinygs/dataloader/nvtx_dl.h"

namespace tinygs {

template<typename T>
class BoundedBlockingQueue {
public:
    explicit BoundedBlockingQueue(size_t capacity) : cap_(capacity) {
        if (cap_ == 0) throw std::invalid_argument("capacity must be > 0");
    }
    BoundedBlockingQueue(const BoundedBlockingQueue&) = delete;
    BoundedBlockingQueue& operator=(const BoundedBlockingQueue&) = delete;
    
    /// @brief Close the queue: reject new pushes; pop returns false when queue is exhausted
    void close() {
        std::scoped_lock lk(m_);
        closed_ = true;
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    void clear() {
        std::scoped_lock lk(m_);
        q_.clear();
        not_full_.notify_all();
    }

    /// @brief Push element to queue, returns false if cancelled or queue is closed
    bool push(T value, std::stop_token st = {}) {
        std::unique_lock lk(m_);
        auto can_push = [&] { return closed_ || q_.size() < cap_; };
        if (!not_full_.wait(lk, st, can_push)) return false; // cancelled
        if (closed_) return false;                           // closed
        q_.push_back(std::move(value));
        lk.unlock();
        not_empty_.notify_one();
        return true;
    }

    /// @brief Pop element from queue, returns false if cancelled or queue is closed and empty
    bool pop(T& out, std::stop_token st = {}) {
        std::unique_lock lk(m_);
        auto can_pop = [&] { return closed_ || !q_.empty(); };
        if (!not_empty_.wait(lk, st, can_pop)) return false; // cancelled
        if (q_.empty()) return false;                        // closed and empty
        out = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        not_full_.notify_one();
        return true;
    }

    size_t size() const {
        std::scoped_lock lk(m_);
        return q_.size();
    }

    bool closed() const {
        std::scoped_lock lk(m_);
        return closed_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable_any not_full_;
    std::condition_variable_any not_empty_;
    std::deque<T> q_;
    size_t cap_;
    bool closed_ = false;
};

/// @brief Implementation struct for AsyncDataLoader (PIMPL idiom)
struct AsyncDataLoader::Impl {
  // Simple CUDA buffer for internal use
  struct CudaDeviceBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    void resize(size_t new_size) {
      if (new_size > size) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
        CUDA_CHECK_THROW(cudaMalloc(&ptr, new_size * sizeof(float)));
        size = new_size;
      }
    }
    float* data() { return static_cast<float*>(ptr); }
    ~CudaDeviceBuffer() { if (ptr) cudaFree(ptr); }
  };
  CudaDeviceBuffer gpu_memory;              ///< GPU buffer for data storage
  pcg32 rng;                                ///< Random number generator
  uint32_t rngseed = 0;                     ///< Seed for random number generator
  std::vector<size_t> permutation;          ///< Current permutation of dataset indices
  size_t current_index;                     ///< Current position in the permutation
  uint32_t prefetch_factor = 4;             ///< Prefetch factor for prefetching data
  DataType data_type = DataType::Float32;   ///< Data type for GPU storage

  std::jthread prefetch_thread;             ///< Thread for prefetching data
  std::unique_ptr<BoundedBlockingQueue<std::pair<GPUBatchInputOutput, uint32_t>>> data_queue;
  std::unique_ptr<BoundedBlockingQueue<uint32_t>> index_queue;
  cudaStream_t prefetch_stream;             ///< CUDA stream for prefetching

  int last_using_buffer_idx = -1;  ///< Last used buffer index to return to index queue

  // Thread safety and error handling
  mutable std::mutex state_mutex_;          ///< Protects shared state (permutation, current_index, rng)
  std::atomic<bool> error_occurred_{false}; ///< Error flag for background thread
  std::string error_message_;               ///< Error message from background thread

  /// @brief Constructor
  Impl() : current_index(0), prefetch_stream(nullptr) { rng.seed(rngseed); }

  /// @brief Destructor - ensures clean shutdown
  ~Impl() {
    shutdown();
  }

  /// @brief Check if prefetch thread has started
  bool has_start() const noexcept {
    return prefetch_thread.joinable();
  }

  /// @brief Start the prefetch thread
  void start(DataLoaderBase& loader, DatasetBase& dataset, DataType data_type) {
    this->data_type = data_type;
    data_queue = std::make_unique<BoundedBlockingQueue<std::pair<GPUBatchInputOutput, uint32_t>>>(prefetch_factor);
    index_queue = std::make_unique<BoundedBlockingQueue<uint32_t>>(prefetch_factor);
    // Preallocate ring buffer to maximum dataset image stride to avoid future reallocations
    auto max_stride = dataset.image_shape().padded_size();
    gpu_memory.resize(max_stride * prefetch_factor);
    index_queue->clear();
    data_queue->clear();
    
    prefetch_thread = std::jthread([&loader, &dataset, this](std::stop_token st) {
      try {
        CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking));
        size_t total_fetched = 0;
        while (!st.stop_requested()) {
          if (!this->prefetch_work(loader, dataset, total_fetched, st)) {
            break; // Normal exit due to queue closure or cancellation
          }
          total_fetched += 1;
        }
      } catch (const std::exception& e) {
        // Handle errors in background thread
        std::lock_guard<std::mutex> lock(state_mutex_);
        error_occurred_ = true;
        error_message_ = e.what();
        // Close queues to notify main thread
        if (data_queue) data_queue->close();
        if (index_queue) index_queue->close();
      }

      // Cleanup CUDA stream
      if (prefetch_stream) {
        CUDA_CHECK_PRINT(cudaStreamDestroy(prefetch_stream));
        prefetch_stream = nullptr;
      }
      log_info("Prefetch thread exited.");
    });

    // Prime index queue with initial buffer indices
    for (uint32_t i = 0; i < prefetch_factor; ++i) {
      if (!index_queue->push(i)) {
        throw std::runtime_error("Failed to initialize index queue");
      }
    }
  }

  /// @brief Gracefully shutdown the prefetch thread
  void shutdown() {
    if (has_start()) {
      // Close queues first to unblock any waiting operations
      if (data_queue) data_queue->close();
      if (index_queue) index_queue->close();

      // Request thread to stop and wait for it
      if (prefetch_thread.joinable()) {
        prefetch_thread.request_stop();
        prefetch_thread.join();
      }

      data_queue->clear();
      index_queue->clear();
    }
  }

  /// @brief Check for errors from background thread
  void check_background_error() {
    if (error_occurred_.load()) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!error_message_.empty()) {
        throw std::runtime_error("Background thread error: " + error_message_);
      } else {
        throw std::runtime_error("Unknown background thread error");
      }
    }
  }

  /// @brief Transfer data from dataset to gpu_memory (with improved error handling)
  bool prefetch_work(DataLoaderBase& base, DatasetBase& dataset, size_t total_fetched, std::stop_token st) {
    DL_RANGE_SCOPE_LIT("prefetch_work", ::dl_nvtx::C_BLUE, ::dl_nvtx::catPrefetch(), total_fetched);
    // Thread-safe access to permutation state
    size_t perm_idx;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (current_index >= permutation.size()) {
        generate_permutation_unsafe(dataset.size());
      }
      perm_idx = permutation[current_index];
      current_index++;
    }

    // Get buffer index from queue
    uint32_t buffer_idx;
    if (!index_queue->pop(buffer_idx, st)) {
      return false; // Queue closed or cancelled
    }

    // Load data from dataset
    Data host_data = dataset[perm_idx];
    
    // Prepare GPU batch input
    GPUBatchInput gpu_input;
    // Keep input resolution consistent with current output shape
    gpu_input.height = m_output_shape.height;
    gpu_input.width = m_output_shape.width;
    gpu_input.near = 0.1f; // Default near plane
    gpu_input.far = 100.0f; // Default far plane
    gpu_input.K = host_data.K;
    gpu_input.w2c = host_data.w2c;
    gpu_input.timestamp = host_data.timestamp;

    // Create GPU image structure
    Image gpu_image;
    gpu_image.shape = m_output_shape;
    gpu_image.data_type = data_type;
    // Use maximum stride for per-buffer segment to avoid overlap after resolution increases
    const size_t stride = dataset.image_shape().padded_size();
    gpu_image.data = this->gpu_memory.data() + stride * buffer_idx;

    // Transfer data from host to GPU using the provided CUDA stream
    base.transfer_gpu(prefetch_stream, gpu_image, host_data.image);

    // Prepare GPU batch output
    GPUBatchOutput gpu_output;
    gpu_output.image = Image{gpu_image.shape, gpu_image.data_type, gpu_image.data};

    GPUBatchInputOutput pld{gpu_input, gpu_output};

    // Push result to data queue
    if (!data_queue->push(std::make_pair(pld, buffer_idx), st)) {
      return false; // Queue closed or cancelled
    }
    log_debug("Prefetched batch {} (perm_idx={}, buffer_idx={})", total_fetched, perm_idx, buffer_idx);
    return true;
  }

  /// @brief Generate a new random permutation of dataset indices (unsafe - caller must hold lock)
  void generate_permutation_unsafe(size_t dataset_size) {
    permutation.resize(dataset_size);
    
    // Initialize permutation with sequential indices
    for (size_t i = 0; i < dataset_size; ++i) {
      permutation[i] = i;
    }
    
    // Fisher-Yates shuffle using our RNG
    for (size_t i = dataset_size - 1; i > 0; --i) {
      size_t j = rng.next_uint(i + 1);
      std::swap(permutation[i], permutation[j]);
    }
    
    // Reset current index to start of new permutation
    current_index = 0;
  }

  /// @brief Thread-safe version of generate_permutation
  void generate_permutation(size_t dataset_size) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    generate_permutation_unsafe(dataset_size);
  }

  ImageShape m_output_shape;
};

AsyncDataLoader::AsyncDataLoader(std::shared_ptr<DatasetBase> dataset) 
  : DataLoaderBase(dataset), m_impl(std::make_unique<Impl>()) {
  m_impl->generate_permutation(m_dataset->size());
}

AsyncDataLoader::~AsyncDataLoader() {
  // Explicit cleanup to ensure proper shutdown order
  if (m_impl) {
    m_impl->shutdown();
  }
}

GPUBatchInputOutput AsyncDataLoader::next() {
  DL_FUNC_RANGE();
  
  // Check for background thread errors
  m_impl->check_background_error();
  
  if (!m_impl->has_start()) {
    reset();
  } else {
    if (m_impl->last_using_buffer_idx >= 0) {
      if (!m_impl->index_queue->push((uint32_t) m_impl->last_using_buffer_idx)) {
        throw std::runtime_error("Failed to return buffer index to queue");
      }
    }
  }

  // Pop a prefetched batch from the queue
  std::pair<GPUBatchInputOutput, uint32_t> pld;
  if (!m_impl->data_queue->pop(pld)) {
    // Check for background thread errors before throwing
    m_impl->check_background_error();
    throw std::runtime_error("Data queue closed or cancelled");
  }
  
  m_impl->last_using_buffer_idx = pld.second;
  return pld.first;
}

void AsyncDataLoader::set_params(const json &params) {
  if (params.contains("seed")) {
    std::lock_guard<std::mutex> lock(m_impl->state_mutex_);
    m_impl->rngseed = params["seed"].get<uint64_t>();
    m_impl->rng.seed(m_impl->rngseed);
  }

  DataLoaderBase::set_params(params);
}

json AsyncDataLoader::get_params() const {
  std::lock_guard<std::mutex> lock(m_impl->state_mutex_);
  json params = DataLoaderBase::get_params();
  params["type"] = "async";
  params["seed"] = m_impl->rngseed;
  return params;
}

void AsyncDataLoader::reset() {
  // Ensure base preallocations (e.g., device scratch buffer) happen once
  DataLoaderBase::reset();
  // Stop current prefetching thread and queues
  m_impl->shutdown();

  // Update impl output shape to match the latest dataloader output shape
  {
    std::lock_guard<std::mutex> lock(m_impl->state_mutex_);
    m_impl->m_output_shape = m_output_shape;
    // Clear previous error state and buffer index
    m_impl->error_occurred_ = false;
    m_impl->error_message_.clear();
    m_impl->last_using_buffer_idx = -1;
  }

  // Regenerate dataset permutation
  m_impl->generate_permutation(m_dataset->size());

  // Relaunch prefetch thread and prime index queue
  m_impl->start(*this, *m_dataset, m_params.data_type);
}

} // namespace tinygs