#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "tinygs/platform/backend_error.hpp"

namespace tinygs {

/// Result wrapper used by runtime factory and allocation APIs.
template <typename T>
class Result {
public:
  Result() = default;
  explicit Result(BackendError error) : m_error(std::move(error)) {}
  Result(std::shared_ptr<T> value, BackendError error)
      : m_value(std::move(value)), m_error(std::move(error)) {}

  static Result<T> success(
      std::shared_ptr<T> value,
      BackendType backend,
      std::string operation = {}) {
    return Result<T>(
        std::move(value),
        backend_success(backend, std::move(operation)));
  }

  static Result<T> failure(BackendError error) {
    return Result<T>(std::shared_ptr<T>{}, std::move(error));
  }

  bool ok() const noexcept { return m_error.ok(); }
  const BackendError& error() const noexcept { return m_error; }
  const std::shared_ptr<T>& value() const noexcept { return m_value; }

private:
  std::shared_ptr<T> m_value{};
  BackendError m_error{};
};

/// Backend feature flags and device properties.
struct CapabilityProfile {
  bool supports_queues = true;
  bool supports_events = true;
  bool supports_device_buffers = true;
  bool supports_host_visible_buffers = false;
  bool supports_unified_memory = false;
  bool supports_interop_buffers = false;
  bool supports_graph_capture = false;

  uint32_t compute_capability = 0;
  size_t total_global_memory_bytes = 0;
};

/// Queue creation parameters.
struct QueueDesc {
  bool non_blocking = true;
  std::string debug_name;
};

/// Event creation parameters.
struct EventDesc {
  bool disable_timing = true;
  std::string debug_name;
};

/// Buffer residency and ownership class.
enum class BufferMemoryClass : uint8_t {
  Device = 0,
  Unified = 1,
  HostPinned = 2,
};

/// Allowed host-side access for a buffer allocation.
enum class BufferHostAccess : uint8_t {
  None = 0,
  Read = 1,
  Write = 2,
  ReadWrite = 3,
};

/// External interop mode for a buffer allocation.
enum class BufferInteropMode : uint8_t {
  None = 0,
  External = 1,
};

/// Buffer allocation descriptor.
struct BufferDesc {
  size_t size_bytes = 0;
  size_t alignment = 256;
  BufferMemoryClass memory_class = BufferMemoryClass::Device;
  BufferHostAccess host_access = BufferHostAccess::None;
  BufferInteropMode interop_mode = BufferInteropMode::None;
  std::string debug_name;
};

/// Device-to-device copy region.
struct CopyRegion {
  size_t size_bytes = 0;
  size_t dst_offset = 0;
  size_t src_offset = 0;
};

/// Host-to-buffer or buffer-to-host copy region.
struct BufferTransferRegion {
  size_t size_bytes = 0;
  size_t buffer_offset = 0;
};

/// Opaque execution queue handle.
class BackendQueue {
public:
  virtual ~BackendQueue() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual void* native_handle() const noexcept = 0;
};

/// Opaque synchronization event handle.
class BackendEvent {
public:
  virtual ~BackendEvent() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual void* native_handle() const noexcept = 0;
};

/// Opaque backend-owned buffer handle.
class BackendBuffer {
public:
  virtual ~BackendBuffer() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual size_t size_bytes() const noexcept = 0;
  virtual const BufferDesc& desc() const noexcept = 0;
  virtual void* data() const noexcept = 0;
  virtual void* native_handle() const noexcept = 0;
};

/// Backend-neutral runtime contract for queues, buffers, and copies.
class BackendRuntime {
public:
  virtual ~BackendRuntime() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual CapabilityProfile capability_profile() const = 0;

  virtual Result<BackendQueue> create_queue(const QueueDesc& desc) = 0;
  virtual Result<BackendEvent> create_event(const EventDesc& desc) = 0;
  virtual Result<BackendBuffer> create_buffer(const BufferDesc& desc) = 0;

  virtual BackendError record_event(
      const std::shared_ptr<BackendQueue>& queue,
      const std::shared_ptr<BackendEvent>& event) = 0;
  virtual BackendError wait_event(
      const std::shared_ptr<BackendQueue>& queue,
      const std::shared_ptr<BackendEvent>& event) = 0;

  virtual BackendError synchronize_queue(const std::shared_ptr<BackendQueue>& queue) = 0;
  virtual BackendError synchronize_event(const std::shared_ptr<BackendEvent>& event) = 0;
  virtual BackendError synchronize_device() = 0;

  /// Enqueue a device-to-device copy.
  virtual BackendError copy_buffer_async(
      const std::shared_ptr<BackendQueue>& queue,
      const std::shared_ptr<BackendBuffer>& dst,
      const std::shared_ptr<BackendBuffer>& src,
      const CopyRegion& region) = 0;

  /// Enqueue a host-to-buffer copy.
  virtual BackendError copy_from_host_async(
      const std::shared_ptr<BackendQueue>& queue,
      const std::shared_ptr<BackendBuffer>& dst,
      const void* src,
      const BufferTransferRegion& region) = 0;

  /// Enqueue a buffer-to-host copy.
  virtual BackendError copy_to_host_async(
      const std::shared_ptr<BackendQueue>& queue,
      void* dst,
      const std::shared_ptr<BackendBuffer>& src,
      const BufferTransferRegion& region) = 0;

  /// Enqueue a raw device-pointer to host copy.
  virtual BackendError copy_device_to_host_async(
      const std::shared_ptr<BackendQueue>& queue,
      void* dst,
      const void* src,
      size_t size_bytes) = 0;

  /// Enqueue a host-to-raw-device-pointer copy.
  virtual BackendError copy_host_to_device_async(
      const std::shared_ptr<BackendQueue>& queue,
      void* dst,
      const void* src,
      size_t size_bytes) = 0;

  /// Enqueue a byte fill over a buffer region.
  virtual BackendError fill_buffer_async(
      const std::shared_ptr<BackendQueue>& queue,
      const std::shared_ptr<BackendBuffer>& buffer,
      uint8_t value,
      size_t offset,
      size_t size_bytes) = 0;

  /// Enqueue a byte fill over the entire buffer.
  BackendError fill_buffer_async(
      const std::shared_ptr<BackendQueue>& queue,
      const std::shared_ptr<BackendBuffer>& buffer,
      uint8_t value) {
    if (!buffer) {
      return backend_error(
          backend_type(),
          BackendErrorCode::InvalidArgument,
          "fill_buffer_async",
          "buffer must not be null");
    }
    return fill_buffer_async(queue, buffer, value, 0, buffer->size_bytes());
  }
};

}  // namespace tinygs
