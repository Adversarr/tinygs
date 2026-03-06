#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "tinygs/cuda/common_host.hpp"
#include "tinygs/platform/runtime.hpp"

namespace tinygs {

namespace {

inline bool is_power_of_two(size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

inline std::string normalize_operation(const std::string& operation) {
  if (operation.empty()) {
    return "unspecified_operation";
  }
  return operation;
}

inline bool add_overflows(size_t lhs, size_t rhs) noexcept {
  return lhs > std::numeric_limits<size_t>::max() - rhs;
}

BackendErrorCode map_cuda_error_code(cudaError_t error) {
  switch (error) {
    case cudaSuccess:
      return BackendErrorCode::Success;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevice:
    case cudaErrorInvalidResourceHandle:
    case cudaErrorInvalidMemcpyDirection:
      return BackendErrorCode::InvalidArgument;
    case cudaErrorNotSupported:
      return BackendErrorCode::Unsupported;
    case cudaErrorMemoryAllocation:
      return BackendErrorCode::OutOfMemory;
    case cudaErrorLaunchTimeout:
    case cudaErrorNotReady:
      return BackendErrorCode::Timeout;
    case cudaErrorDeviceUninitialized:
      return BackendErrorCode::DeviceLost;
    case cudaErrorUnknown:
      return BackendErrorCode::UnknownFailure;
    default:
      return BackendErrorCode::RuntimeFailure;
  }
}

BackendError cuda_status(cudaError_t error, const std::string& operation) {
  const std::string op_name = normalize_operation(operation);
  if (error == cudaSuccess) {
    return backend_success(BackendType::Cuda, op_name);
  }
  return backend_error(BackendType::Cuda,
                       map_cuda_error_code(error),
                       op_name,
                       cudaGetErrorString(error));
}

class CudaQueue final : public BackendQueue {
public:
  CudaQueue(int device, cudaStream_t stream) : m_device(device), m_stream(stream) {}

  ~CudaQueue() override {
    if (m_stream != nullptr) {
      const cudaError_t error = cudaStreamDestroy(m_stream);
      if (error != cudaSuccess) {
        log_error("cudaStreamDestroy failed: {}", cudaGetErrorString(error));
      }
    }
  }

  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override { return reinterpret_cast<void*>(m_stream); }

  cudaStream_t stream() const noexcept { return m_stream; }

private:
  int m_device = 0;
  cudaStream_t m_stream = nullptr;
};

class CudaEvent final : public BackendEvent {
public:
  CudaEvent(int device, cudaEvent_t event) : m_device(device), m_event(event) {}

  ~CudaEvent() override {
    if (m_event != nullptr) {
      const cudaError_t error = cudaEventDestroy(m_event);
      if (error != cudaSuccess) {
        log_error("cudaEventDestroy failed: {}", cudaGetErrorString(error));
      }
    }
  }

  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override { return reinterpret_cast<void*>(m_event); }

  cudaEvent_t event() const noexcept { return m_event; }

private:
  int m_device = 0;
  cudaEvent_t m_event = nullptr;
};

class CudaBuffer final : public BackendBuffer {
public:
  CudaBuffer(int device, BufferDesc desc, void* data) : m_device(device), m_desc(std::move(desc)), m_data(data) {}

  ~CudaBuffer() override {
    if (m_data == nullptr) {
      return;
    }

    cudaError_t error = cudaSuccess;
    switch (m_desc.memory_class) {
      case BufferMemoryClass::Device:
      case BufferMemoryClass::Unified:
        error = cudaFree(m_data);
        break;
      case BufferMemoryClass::HostPinned:
        error = cudaFreeHost(m_data);
        break;
      default:
        error = cudaErrorInvalidValue;
        break;
    }
    if (error != cudaSuccess) {
      log_error("Buffer free failed: {}", cudaGetErrorString(error));
    }
  }

  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return m_device; }
  size_t size_bytes() const noexcept override { return m_desc.size_bytes; }
  const BufferDesc& desc() const noexcept override { return m_desc; }
  void* data() const noexcept override { return m_data; }
  void* native_handle() const noexcept override { return m_data; }

private:
  int m_device = 0;
  BufferDesc m_desc;
  void* m_data = nullptr;
};

class CudaRuntime final : public BackendRuntime {
public:
  explicit CudaRuntime(int device) : m_device(device) {
    if (m_device < 0) {
      throw std::runtime_error("backend.device must be >= 0 for CUDA runtime.");
    }

    const int device_count = cuda_device_count();
    if (m_device >= device_count) {
      throw std::runtime_error("Requested CUDA device index out of range.");
    }

    set_cuda_device(m_device);

    cudaDeviceProp props{};
    CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, m_device));

    m_capability_profile.supports_queues = true;
    m_capability_profile.supports_events = true;
    m_capability_profile.supports_device_buffers = true;
    m_capability_profile.supports_unified_memory = props.managedMemory != 0;
    m_capability_profile.supports_host_visible_buffers = props.managedMemory != 0;
    m_capability_profile.supports_interop_buffers = false;
    m_capability_profile.supports_graph_capture = false;
    m_capability_profile.compute_capability =
        static_cast<uint32_t>(props.major * 10 + props.minor);
    m_capability_profile.total_global_memory_bytes =
        static_cast<size_t>(props.totalGlobalMem);
  }

  ~CudaRuntime() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return m_device; }
  CapabilityProfile capability_profile() const override { return m_capability_profile; }

  Result<BackendQueue> create_queue(const QueueDesc& desc) override {
    constexpr const char* operation = "create_queue";
    const BackendError device_status = ensure_device(operation);
    if (!device_status.ok()) {
      return Result<BackendQueue>::failure(device_status);
    }

    cudaStream_t stream = nullptr;
    const unsigned int flags = desc.non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
    const cudaError_t error = cudaStreamCreateWithFlags(&stream, flags);
    if (error != cudaSuccess) {
      return Result<BackendQueue>::failure(cuda_status(error, operation));
    }

    return Result<BackendQueue>::success(
        std::make_shared<CudaQueue>(m_device, stream),
        BackendType::Cuda,
        operation);
  }

  Result<BackendEvent> create_event(const EventDesc& desc) override {
    constexpr const char* operation = "create_event";
    const BackendError device_status = ensure_device(operation);
    if (!device_status.ok()) {
      return Result<BackendEvent>::failure(device_status);
    }

    cudaEvent_t event = nullptr;
    const unsigned int flags = desc.disable_timing ? cudaEventDisableTiming : cudaEventDefault;
    const cudaError_t error = cudaEventCreateWithFlags(&event, flags);
    if (error != cudaSuccess) {
      return Result<BackendEvent>::failure(cuda_status(error, operation));
    }

    return Result<BackendEvent>::success(
        std::make_shared<CudaEvent>(m_device, event),
        BackendType::Cuda,
        operation);
  }

  Result<BackendBuffer> create_buffer(const BufferDesc& desc) override {
    constexpr const char* operation = "create_buffer";
    if (desc.size_bytes == 0) {
      return Result<BackendBuffer>::failure(
          backend_error(BackendType::Cuda,
                        BackendErrorCode::InvalidArgument,
                        operation,
                        "buffer size_bytes must be > 0"));
    }
    if (!is_power_of_two(desc.alignment)) {
      return Result<BackendBuffer>::failure(
          backend_error(BackendType::Cuda,
                        BackendErrorCode::InvalidArgument,
                        operation,
                        "buffer alignment must be a power of two"));
    }
    if (desc.interop_mode != BufferInteropMode::None) {
      return Result<BackendBuffer>::failure(
          backend_error(BackendType::Cuda,
                        BackendErrorCode::Unsupported,
                        operation,
                        "interop buffers are not implemented for CUDA runtime yet"));
    }
    if (desc.memory_class == BufferMemoryClass::Device &&
        desc.host_access != BufferHostAccess::None) {
      return Result<BackendBuffer>::failure(
          backend_error(BackendType::Cuda,
                        BackendErrorCode::Unsupported,
                        operation,
                        "device buffers do not support host access in CUDA runtime"));
    }

    const BackendError device_status = ensure_device(operation);
    if (!device_status.ok()) {
      return Result<BackendBuffer>::failure(device_status);
    }

    void* data = nullptr;
    cudaError_t error = cudaSuccess;
    switch (desc.memory_class) {
      case BufferMemoryClass::Device:
        error = cudaMalloc(&data, desc.size_bytes);
        break;
      case BufferMemoryClass::Unified:
        error = cudaMallocManaged(&data, desc.size_bytes);
        break;
      case BufferMemoryClass::HostPinned:
        error = cudaMallocHost(&data, desc.size_bytes);
        break;
      default:
        error = cudaErrorNotSupported;
        break;
    }
    if (error != cudaSuccess) {
      return Result<BackendBuffer>::failure(cuda_status(error, operation));
    }

    BufferDesc normalized_desc = desc;
    if (normalized_desc.memory_class == BufferMemoryClass::Unified &&
        normalized_desc.host_access == BufferHostAccess::None) {
      normalized_desc.host_access = BufferHostAccess::ReadWrite;
    }
    return Result<BackendBuffer>::success(
        std::make_shared<CudaBuffer>(m_device, std::move(normalized_desc), data),
        BackendType::Cuda,
        operation);
  }

  BackendError record_event(const std::shared_ptr<BackendQueue>& queue,
                            const std::shared_ptr<BackendEvent>& event) override {
    constexpr const char* operation = "record_event";
    std::shared_ptr<CudaQueue> cuda_queue;
    std::shared_ptr<CudaEvent> cuda_event;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    status = require_event(event, &cuda_event, operation);
    if (!status.ok()) {
      return status;
    }

    const cudaError_t error = cudaEventRecord(cuda_event->event(), cuda_queue->stream());
    return cuda_status(error, operation);
  }

  BackendError wait_event(const std::shared_ptr<BackendQueue>& queue,
                          const std::shared_ptr<BackendEvent>& event) override {
    constexpr const char* operation = "wait_event";
    std::shared_ptr<CudaQueue> cuda_queue;
    std::shared_ptr<CudaEvent> cuda_event;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    status = require_event(event, &cuda_event, operation);
    if (!status.ok()) {
      return status;
    }

    const cudaError_t error = cudaStreamWaitEvent(cuda_queue->stream(), cuda_event->event(), 0);
    return cuda_status(error, operation);
  }

  BackendError synchronize_queue(const std::shared_ptr<BackendQueue>& queue) override {
    constexpr const char* operation = "synchronize_queue";
    std::shared_ptr<CudaQueue> cuda_queue;
    const BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    const cudaError_t error = cudaStreamSynchronize(cuda_queue->stream());
    return cuda_status(error, operation);
  }

  BackendError synchronize_event(const std::shared_ptr<BackendEvent>& event) override {
    constexpr const char* operation = "synchronize_event";
    std::shared_ptr<CudaEvent> cuda_event;
    const BackendError status = require_event(event, &cuda_event, operation);
    if (!status.ok()) {
      return status;
    }
    const cudaError_t error = cudaEventSynchronize(cuda_event->event());
    return cuda_status(error, operation);
  }

  BackendError synchronize_device() override {
    constexpr const char* operation = "synchronize_device";
    const BackendError status = ensure_device(operation);
    if (!status.ok()) {
      return status;
    }
    const cudaError_t error = cudaDeviceSynchronize();
    return cuda_status(error, operation);
  }

  BackendError copy_buffer_async(const std::shared_ptr<BackendQueue>& queue,
                                 const std::shared_ptr<BackendBuffer>& dst,
                                 const std::shared_ptr<BackendBuffer>& src,
                                 const CopyRegion& region) override {
    constexpr const char* operation = "copy_buffer_async";
    std::shared_ptr<CudaQueue> cuda_queue;
    std::shared_ptr<CudaBuffer> cuda_dst;
    std::shared_ptr<CudaBuffer> cuda_src;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    status = require_buffer(dst, &cuda_dst, operation);
    if (!status.ok()) {
      return status;
    }
    status = require_buffer(src, &cuda_src, operation);
    if (!status.ok()) {
      return status;
    }

    status = validate_copy_region(cuda_dst, cuda_src, region, operation);
    if (!status.ok()) {
      return status;
    }
    if (region.size_bytes == 0) {
      return backend_success(BackendType::Cuda, operation);
    }

    const auto* src_bytes = static_cast<const char*>(cuda_src->data()) + region.src_offset;
    auto* dst_bytes = static_cast<char*>(cuda_dst->data()) + region.dst_offset;
    const cudaError_t error =
        cudaMemcpyAsync(dst_bytes,
                        src_bytes,
                        region.size_bytes,
                        cudaMemcpyDeviceToDevice,
                        cuda_queue->stream());
    return cuda_status(error, operation);
  }

  BackendError copy_from_host_async(const std::shared_ptr<BackendQueue>& queue,
                                    const std::shared_ptr<BackendBuffer>& dst,
                                    const void* src,
                                    const BufferTransferRegion& region) override {
    constexpr const char* operation = "copy_from_host_async";
    std::shared_ptr<CudaQueue> cuda_queue;
    std::shared_ptr<CudaBuffer> cuda_dst;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    status = require_buffer(dst, &cuda_dst, operation);
    if (!status.ok()) {
      return status;
    }
    if (region.size_bytes == 0) {
      return backend_success(BackendType::Cuda, operation);
    }
    if (src == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "src must not be null when size_bytes > 0");
    }
    if (add_overflows(region.buffer_offset, region.size_bytes) ||
        region.buffer_offset + region.size_bytes > cuda_dst->size_bytes()) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "destination copy range exceeds buffer size");
    }

    auto* dst_bytes = static_cast<char*>(cuda_dst->data()) + region.buffer_offset;
    const cudaError_t error =
        cudaMemcpyAsync(dst_bytes,
                        src,
                        region.size_bytes,
                        cudaMemcpyHostToDevice,
                        cuda_queue->stream());
    return cuda_status(error, operation);
  }

  BackendError copy_to_host_async(const std::shared_ptr<BackendQueue>& queue,
                                  void* dst,
                                  const std::shared_ptr<BackendBuffer>& src,
                                  const BufferTransferRegion& region) override {
    constexpr const char* operation = "copy_to_host_async";
    std::shared_ptr<CudaQueue> cuda_queue;
    std::shared_ptr<CudaBuffer> cuda_src;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    status = require_buffer(src, &cuda_src, operation);
    if (!status.ok()) {
      return status;
    }
    if (region.size_bytes == 0) {
      return backend_success(BackendType::Cuda, operation);
    }
    if (dst == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "dst must not be null when size_bytes > 0");
    }
    if (add_overflows(region.buffer_offset, region.size_bytes) ||
        region.buffer_offset + region.size_bytes > cuda_src->size_bytes()) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "source copy range exceeds buffer size");
    }

    const auto* src_bytes = static_cast<const char*>(cuda_src->data()) + region.buffer_offset;
    const cudaError_t error =
        cudaMemcpyAsync(dst,
                        src_bytes,
                        region.size_bytes,
                        cudaMemcpyDeviceToHost,
                        cuda_queue->stream());
    return cuda_status(error, operation);
  }

  BackendError copy_device_to_host_async(const std::shared_ptr<BackendQueue>& queue,
                                         void* dst,
                                         const void* src,
                                         size_t size_bytes) override {
    constexpr const char* operation = "copy_device_to_host_async";
    std::shared_ptr<CudaQueue> cuda_queue;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    if (size_bytes == 0) {
      return backend_success(BackendType::Cuda, operation);
    }
    if (dst == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "dst must not be null when size_bytes > 0");
    }
    if (src == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "src must not be null when size_bytes > 0");
    }

    const cudaError_t error =
        cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyDeviceToHost, cuda_queue->stream());
    return cuda_status(error, operation);
  }

  BackendError copy_host_to_device_async(const std::shared_ptr<BackendQueue>& queue,
                                          void* dst,
                                          const void* src,
                                          size_t size_bytes) override {
    constexpr const char* operation = "copy_host_to_device_async";
    std::shared_ptr<CudaQueue> cuda_queue;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    if (size_bytes == 0) {
      return backend_success(BackendType::Cuda, operation);
    }
    if (dst == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "dst must not be null when size_bytes > 0");
    }
    if (src == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "src must not be null when size_bytes > 0");
    }

    const cudaError_t error =
        cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyHostToDevice, cuda_queue->stream());
    return cuda_status(error, operation);
  }

  BackendError fill_buffer_async(const std::shared_ptr<BackendQueue>& queue,
                                 const std::shared_ptr<BackendBuffer>& buffer,
                                 uint8_t value,
                                 size_t offset,
                                 size_t size_bytes) override {
    constexpr const char* operation = "fill_buffer_async";
    std::shared_ptr<CudaQueue> cuda_queue;
    std::shared_ptr<CudaBuffer> cuda_buffer;
    BackendError status = require_queue(queue, &cuda_queue, operation);
    if (!status.ok()) {
      return status;
    }
    status = require_buffer(buffer, &cuda_buffer, operation);
    if (!status.ok()) {
      return status;
    }
    if (size_bytes == 0) {
      return backend_success(BackendType::Cuda, operation);
    }
    if (add_overflows(offset, size_bytes) ||
        offset + size_bytes > cuda_buffer->size_bytes()) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "fill range exceeds buffer size");
    }
    auto* dst = static_cast<char*>(cuda_buffer->data()) + offset;
    const cudaError_t error =
        cudaMemsetAsync(dst, static_cast<int>(value), size_bytes, cuda_queue->stream());
    return cuda_status(error, operation);
  }

private:
  BackendError ensure_device(const std::string& operation) const {
    const cudaError_t error = cudaSetDevice(m_device);
    return cuda_status(error, operation);
  }

  BackendError require_queue(const std::shared_ptr<BackendQueue>& queue,
                             std::shared_ptr<CudaQueue>* out_queue,
                             const std::string& operation) const {
    if (queue == nullptr || out_queue == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "queue must not be null");
    }
    if (queue->backend_type() != BackendType::Cuda || queue->device() != m_device) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "queue backend/device mismatch");
    }
    auto typed = std::dynamic_pointer_cast<CudaQueue>(queue);
    if (typed == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "queue handle is not a CUDA queue");
    }
    *out_queue = std::move(typed);
    return backend_success(BackendType::Cuda, operation);
  }

  BackendError require_event(const std::shared_ptr<BackendEvent>& event,
                             std::shared_ptr<CudaEvent>* out_event,
                             const std::string& operation) const {
    if (event == nullptr || out_event == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "event must not be null");
    }
    if (event->backend_type() != BackendType::Cuda || event->device() != m_device) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "event backend/device mismatch");
    }
    auto typed = std::dynamic_pointer_cast<CudaEvent>(event);
    if (typed == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "event handle is not a CUDA event");
    }
    *out_event = std::move(typed);
    return backend_success(BackendType::Cuda, operation);
  }

  BackendError require_buffer(const std::shared_ptr<BackendBuffer>& buffer,
                              std::shared_ptr<CudaBuffer>* out_buffer,
                              const std::string& operation) const {
    if (buffer == nullptr || out_buffer == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "buffer must not be null");
    }
    if (buffer->backend_type() != BackendType::Cuda || buffer->device() != m_device) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "buffer backend/device mismatch");
    }
    auto typed = std::dynamic_pointer_cast<CudaBuffer>(buffer);
    if (typed == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "buffer handle is not a CUDA buffer");
    }
    *out_buffer = std::move(typed);
    return backend_success(BackendType::Cuda, operation);
  }

  BackendError validate_copy_region(const std::shared_ptr<CudaBuffer>& dst,
                                    const std::shared_ptr<CudaBuffer>& src,
                                    const CopyRegion& region,
                                    const std::string& operation) const {
    if (dst == nullptr || src == nullptr) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "copy buffers must not be null");
    }
    if (add_overflows(region.dst_offset, region.size_bytes) ||
        region.dst_offset + region.size_bytes > dst->size_bytes()) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "destination copy range exceeds buffer size");
    }
    if (add_overflows(region.src_offset, region.size_bytes) ||
        region.src_offset + region.size_bytes > src->size_bytes()) {
      return backend_error(BackendType::Cuda,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "source copy range exceeds buffer size");
    }
    return backend_success(BackendType::Cuda, operation);
  }

private:
  int m_device = 0;
  CapabilityProfile m_capability_profile;
};

}  // namespace

Result<BackendRuntime> create_cuda_backend_runtime(int device) {
  constexpr const char* operation = "create_cuda_backend_runtime";
  try {
    return Result<BackendRuntime>::success(
        std::make_shared<CudaRuntime>(device),
        BackendType::Cuda,
        operation);
  } catch (const std::exception& e) {
    return Result<BackendRuntime>::failure(
        backend_error(BackendType::Cuda,
                      BackendErrorCode::InvalidArgument,
                      operation,
                      e.what()));
  }
}

}  // namespace tinygs
