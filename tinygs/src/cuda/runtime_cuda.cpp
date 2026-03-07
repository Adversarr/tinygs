#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "tinygs/cuda/common_host.hpp"
#include "tinygs/platform/runtime.hpp"

namespace tinygs {

namespace {

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

BackendError cuda_status(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) {
    return backend_success(BackendType::Cuda, operation);
  }
  return backend_error(BackendType::Cuda,
                       map_cuda_error_code(error),
                       operation,
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
  CudaBuffer(int device, BufferDesc desc, void* data)
      : m_device(device), m_desc(std::move(desc)), m_data(data) {}

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

// Helper: set current CUDA device and return status.
inline BackendError ensure_cuda_device(int device) {
  return cuda_status(cudaSetDevice(device), "ensure_device");
}

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

protected:
  // ---- do_* primitives (all args pre-validated by NVI base) ----

  Result<BackendQueue> do_create_queue(const QueueDesc& desc) override {
    constexpr const char* op = "create_queue";
    if (auto s = ensure_cuda_device(m_device); !s.ok()) {
      return Result<BackendQueue>::failure(s);
    }
    cudaStream_t stream = nullptr;
    const unsigned int flags = desc.non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
    const cudaError_t error = cudaStreamCreateWithFlags(&stream, flags);
    if (error != cudaSuccess) {
      return Result<BackendQueue>::failure(cuda_status(error, op));
    }
    return Result<BackendQueue>::success(
        std::make_shared<CudaQueue>(m_device, stream), BackendType::Cuda, op);
  }

  Result<BackendEvent> do_create_event(const EventDesc& desc) override {
    constexpr const char* op = "create_event";
    if (auto s = ensure_cuda_device(m_device); !s.ok()) {
      return Result<BackendEvent>::failure(s);
    }
    cudaEvent_t event = nullptr;
    const unsigned int flags = desc.disable_timing ? cudaEventDisableTiming : cudaEventDefault;
    const cudaError_t error = cudaEventCreateWithFlags(&event, flags);
    if (error != cudaSuccess) {
      return Result<BackendEvent>::failure(cuda_status(error, op));
    }
    return Result<BackendEvent>::success(
        std::make_shared<CudaEvent>(m_device, event), BackendType::Cuda, op);
  }

  Result<BackendBuffer> do_create_buffer(const BufferDesc& desc) override {
    constexpr const char* op = "create_buffer";
    if (auto s = ensure_cuda_device(m_device); !s.ok()) {
      return Result<BackendBuffer>::failure(s);
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
      return Result<BackendBuffer>::failure(cuda_status(error, op));
    }
    BufferDesc normalized_desc = desc;
    if (normalized_desc.memory_class == BufferMemoryClass::Unified &&
        normalized_desc.host_access == BufferHostAccess::None) {
      normalized_desc.host_access = BufferHostAccess::ReadWrite;
    }
    return Result<BackendBuffer>::success(
        std::make_shared<CudaBuffer>(m_device, std::move(normalized_desc), data),
        BackendType::Cuda, op);
  }

  BackendError do_record_event(BackendQueue& queue, BackendEvent& event) override {
    auto& cq = static_cast<CudaQueue&>(queue);
    auto& ce = static_cast<CudaEvent&>(event);
    return cuda_status(cudaEventRecord(ce.event(), cq.stream()), "record_event");
  }

  BackendError do_wait_event(BackendQueue& queue, BackendEvent& event) override {
    auto& cq = static_cast<CudaQueue&>(queue);
    auto& ce = static_cast<CudaEvent&>(event);
    return cuda_status(cudaStreamWaitEvent(cq.stream(), ce.event(), 0), "wait_event");
  }

  BackendError do_synchronize_queue(BackendQueue& queue) override {
    auto& cq = static_cast<CudaQueue&>(queue);
    return cuda_status(cudaStreamSynchronize(cq.stream()), "synchronize_queue");
  }

  BackendError do_synchronize_event(BackendEvent& event) override {
    auto& ce = static_cast<CudaEvent&>(event);
    return cuda_status(cudaEventSynchronize(ce.event()), "synchronize_event");
  }

  BackendError do_synchronize_device() override {
    if (auto s = ensure_cuda_device(m_device); !s.ok()) return s;
    return cuda_status(cudaDeviceSynchronize(), "synchronize_device");
  }

  BackendError do_copy_buffer(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    auto stream = static_cast<CudaQueue&>(queue).stream();
    auto* dst_ptr = static_cast<char*>(dst.data()) + dst_offset;
    const auto* src_ptr = static_cast<const char*>(src.data()) + src_offset;
    return cuda_status(
        cudaMemcpyAsync(dst_ptr, src_ptr, size_bytes, cudaMemcpyDeviceToDevice, stream),
        "copy_buffer_async");
  }

  BackendError do_copy_from_host(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      const void* src, size_t size_bytes) override {
    auto stream = static_cast<CudaQueue&>(queue).stream();
    auto* dst_ptr = static_cast<char*>(dst.data()) + dst_offset;
    return cuda_status(
        cudaMemcpyAsync(dst_ptr, src, size_bytes, cudaMemcpyHostToDevice, stream),
        "copy_from_host_async");
  }

  BackendError do_copy_to_host(
      BackendQueue& queue,
      void* dst,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    auto stream = static_cast<CudaQueue&>(queue).stream();
    const auto* src_ptr = static_cast<const char*>(src.data()) + src_offset;
    return cuda_status(
        cudaMemcpyAsync(dst, src_ptr, size_bytes, cudaMemcpyDeviceToHost, stream),
        "copy_to_host_async");
  }

  BackendError do_transfer_raw(
      BackendQueue& queue,
      void* dst, const void* src,
      size_t size_bytes,
      TransferDirection direction) override {
    auto stream = static_cast<CudaQueue&>(queue).stream();
    auto kind = (direction == TransferDirection::HostToDevice)
        ? cudaMemcpyHostToDevice
        : cudaMemcpyDeviceToHost;
    return cuda_status(cudaMemcpyAsync(dst, src, size_bytes, kind, stream), "transfer_raw");
  }

  BackendError do_fill_buffer(
      BackendQueue& queue,
      BackendBuffer& buffer,
      size_t offset, uint8_t value, size_t size_bytes) override {
    auto stream = static_cast<CudaQueue&>(queue).stream();
    auto* dst = static_cast<char*>(buffer.data()) + offset;
    return cuda_status(
        cudaMemsetAsync(dst, static_cast<int>(value), size_bytes, stream),
        "fill_buffer_async");
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
