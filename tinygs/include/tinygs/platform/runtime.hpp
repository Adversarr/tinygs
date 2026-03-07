#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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

/// Raw pointer transfer direction.
enum class TransferDirection : uint8_t {
  HostToDevice = 0,
  DeviceToHost = 1,
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

/// Backend-neutral runtime contract using Non-Virtual Interface (NVI).
///
/// Public methods perform all validation (null checks, bounds, capability, type
/// matching) then delegate to protected `do_*` primitives that backends override.
/// Backends only implement ~13 thin primitives with pre-validated arguments.
class BackendRuntime {
public:
  virtual ~BackendRuntime() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual CapabilityProfile capability_profile() const = 0;

  // ---- Resource creation (NVI wrappers) ----

  Result<BackendQueue> create_queue(const QueueDesc& desc) {
    auto result = do_create_queue(desc);
    if (result.ok()) {
      std::lock_guard<std::mutex> lock(m_queues_mutex);
      m_created_queues.push_back(result.value());
    }
    return result;
  }

  Result<BackendEvent> create_event(const EventDesc& desc) {
    return do_create_event(desc);
  }

  Result<BackendBuffer> create_buffer(const BufferDesc& desc) {
    constexpr const char* op = "create_buffer";
    if (desc.size_bytes == 0) {
      return Result<BackendBuffer>::failure(
          make_error(op, BackendErrorCode::InvalidArgument, "buffer size_bytes must be > 0"));
    }
    if (!is_power_of_two(desc.alignment)) {
      return Result<BackendBuffer>::failure(
          make_error(op, BackendErrorCode::InvalidArgument, "buffer alignment must be a power of two"));
    }
    if (desc.interop_mode != BufferInteropMode::None) {
      return Result<BackendBuffer>::failure(
          make_error(op, BackendErrorCode::Unsupported, "interop buffers are not yet implemented"));
    }
    if (desc.memory_class == BufferMemoryClass::Device &&
        desc.host_access != BufferHostAccess::None) {
      return Result<BackendBuffer>::failure(
          make_error(op, BackendErrorCode::Unsupported, "device buffers do not support host access"));
    }
    return do_create_buffer(desc);
  }

  // ---- Event operations (NVI wrappers) ----

  BackendError record_event(BackendQueue& queue, BackendEvent& event) {
    constexpr const char* op = "record_event";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (auto err = require_event(event, op); !err.ok()) return err;
    return do_record_event(queue, event);
  }

  BackendError wait_event(BackendQueue& queue, BackendEvent& event) {
    constexpr const char* op = "wait_event";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (auto err = require_event(event, op); !err.ok()) return err;
    return do_wait_event(queue, event);
  }

  // ---- Synchronization (NVI wrappers) ----

  BackendError synchronize_queue(BackendQueue& queue) {
    constexpr const char* op = "synchronize_queue";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    return do_synchronize_queue(queue);
  }

  BackendError synchronize_event(BackendEvent& event) {
    constexpr const char* op = "synchronize_event";
    if (auto err = require_event(event, op); !err.ok()) return err;
    return do_synchronize_event(event);
  }

  BackendError synchronize_device() {
    return do_synchronize_device();
  }

  // ---- Copy operations (NVI wrappers) ----

  /// Enqueue a device-to-device copy.
  BackendError copy_buffer_async(
      BackendQueue& queue,
      BackendBuffer& dst,
      BackendBuffer& src,
      const CopyRegion& region) {
    constexpr const char* op = "copy_buffer_async";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (auto err = require_buffer(dst, op, "dst"); !err.ok()) return err;
    if (auto err = require_buffer(src, op, "src"); !err.ok()) return err;
    if (auto err = validate_copy_region(dst, src, region, op); !err.ok()) return err;
    if (region.size_bytes == 0) return make_success(op);
    return do_copy_buffer(queue, dst, region.dst_offset, src, region.src_offset, region.size_bytes);
  }

  /// Enqueue a host-to-buffer copy.
  BackendError copy_from_host_async(
      BackendQueue& queue,
      BackendBuffer& dst,
      const void* src,
      const BufferTransferRegion& region) {
    constexpr const char* op = "copy_from_host_async";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (auto err = require_buffer(dst, op, "dst"); !err.ok()) return err;
    if (region.size_bytes == 0) return make_success(op);
    if (src == nullptr) {
      return make_error(op, BackendErrorCode::InvalidArgument, "src must not be null when size_bytes > 0");
    }
    if (auto err = validate_buffer_region(dst, region, op); !err.ok()) return err;
    return do_copy_from_host(queue, dst, region.buffer_offset, src, region.size_bytes);
  }

  /// Enqueue a buffer-to-host copy.
  BackendError copy_to_host_async(
      BackendQueue& queue,
      void* dst,
      BackendBuffer& src,
      const BufferTransferRegion& region) {
    constexpr const char* op = "copy_to_host_async";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (auto err = require_buffer(src, op, "src"); !err.ok()) return err;
    if (region.size_bytes == 0) return make_success(op);
    if (dst == nullptr) {
      return make_error(op, BackendErrorCode::InvalidArgument, "dst must not be null when size_bytes > 0");
    }
    if (auto err = validate_buffer_region(src, region, op); !err.ok()) return err;
    return do_copy_to_host(queue, dst, src, region.buffer_offset, region.size_bytes);
  }

  /// Enqueue a raw device-pointer to host copy.
  BackendError copy_device_to_host_async(
      BackendQueue& queue,
      void* dst,
      const void* src,
      size_t size_bytes) {
    constexpr const char* op = "copy_device_to_host_async";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (size_bytes == 0) return make_success(op);
    if (dst == nullptr || src == nullptr) {
      return make_error(op, BackendErrorCode::InvalidArgument, "dst and src must not be null when size_bytes > 0");
    }
    return do_transfer_raw(queue, dst, src, size_bytes, TransferDirection::DeviceToHost);
  }

  /// Enqueue a host-to-raw-device-pointer copy.
  BackendError copy_host_to_device_async(
      BackendQueue& queue,
      void* dst,
      const void* src,
      size_t size_bytes) {
    constexpr const char* op = "copy_host_to_device_async";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (size_bytes == 0) return make_success(op);
    if (dst == nullptr || src == nullptr) {
      return make_error(op, BackendErrorCode::InvalidArgument, "dst and src must not be null when size_bytes > 0");
    }
    return do_transfer_raw(queue, dst, src, size_bytes, TransferDirection::HostToDevice);
  }

  // ---- Fill operations (NVI wrappers) ----

  /// Enqueue a byte fill over a buffer region.
  BackendError fill_buffer_async(
      BackendQueue& queue,
      BackendBuffer& buffer,
      uint8_t value,
      size_t offset,
      size_t size_bytes) {
    constexpr const char* op = "fill_buffer_async";
    if (auto err = require_queue(queue, op); !err.ok()) return err;
    if (auto err = require_buffer(buffer, op, "buffer"); !err.ok()) return err;
    if (size_bytes == 0) return make_success(op);
    if (add_overflows(offset, size_bytes) || offset + size_bytes > buffer.size_bytes()) {
      return make_error(op, BackendErrorCode::InvalidArgument, "fill range exceeds buffer size");
    }
    return do_fill_buffer(queue, buffer, offset, value, size_bytes);
  }

  /// Enqueue a byte fill over the entire buffer.
  BackendError fill_buffer_async(
      BackendQueue& queue,
      BackendBuffer& buffer,
      uint8_t value) {
    return fill_buffer_async(queue, buffer, value, 0, buffer.size_bytes());
  }

protected:
  // ---- Backend primitives (override these — arguments are pre-validated) ----

  virtual Result<BackendQueue> do_create_queue(const QueueDesc& desc) = 0;
  virtual Result<BackendEvent> do_create_event(const EventDesc& desc) = 0;
  virtual Result<BackendBuffer> do_create_buffer(const BufferDesc& desc) = 0;

  virtual BackendError do_record_event(BackendQueue& queue, BackendEvent& event) = 0;
  virtual BackendError do_wait_event(BackendQueue& queue, BackendEvent& event) = 0;

  virtual BackendError do_synchronize_queue(BackendQueue& queue) = 0;
  virtual BackendError do_synchronize_event(BackendEvent& event) = 0;

  /// Default: synchronize all live queues. CUDA overrides with cudaDeviceSynchronize.
  virtual BackendError do_synchronize_device() {
    std::lock_guard<std::mutex> lock(m_queues_mutex);
    // Prune expired entries and synchronize live queues.
    auto it = m_created_queues.begin();
    while (it != m_created_queues.end()) {
      auto queue = it->lock();
      if (!queue) {
        it = m_created_queues.erase(it);
        continue;
      }
      auto err = do_synchronize_queue(*queue);
      if (!err.ok()) return err;
      ++it;
    }
    return make_success("synchronize_device");
  }

  virtual BackendError do_copy_buffer(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) = 0;

  virtual BackendError do_copy_from_host(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      const void* src, size_t size_bytes) = 0;

  virtual BackendError do_copy_to_host(
      BackendQueue& queue,
      void* dst,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) = 0;

  virtual BackendError do_transfer_raw(
      BackendQueue& queue,
      void* dst, const void* src,
      size_t size_bytes,
      TransferDirection direction) = 0;

  virtual BackendError do_fill_buffer(
      BackendQueue& queue,
      BackendBuffer& buffer,
      size_t offset, uint8_t value, size_t size_bytes) = 0;

private:
  // ---- Shared validation helpers ----

  static bool is_power_of_two(size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
  }

  static bool add_overflows(size_t lhs, size_t rhs) noexcept {
    return lhs > std::numeric_limits<size_t>::max() - rhs;
  }

  BackendError make_success(const char* op) const {
    return backend_success(backend_type(), op);
  }

  BackendError make_error(const char* op, BackendErrorCode code, const char* msg) const {
    return backend_error(backend_type(), code, op, msg);
  }

  BackendError require_queue(const BackendQueue& queue, const char* op) const {
    if (queue.backend_type() != backend_type() || queue.device() != device()) {
      return make_error(op, BackendErrorCode::InvalidArgument, "queue backend/device mismatch");
    }
    return make_success(op);
  }

  BackendError require_event(const BackendEvent& event, const char* op) const {
    if (event.backend_type() != backend_type() || event.device() != device()) {
      return make_error(op, BackendErrorCode::InvalidArgument, "event backend/device mismatch");
    }
    return make_success(op);
  }

  BackendError require_buffer(
      const BackendBuffer& buffer,
      const char* op,
      const char* name) const {
    if (buffer.backend_type() != backend_type() || buffer.device() != device()) {
      return make_error(op, BackendErrorCode::InvalidArgument,
                        (std::string(name) + " backend/device mismatch").c_str());
    }
    return make_success(op);
  }

  BackendError validate_copy_region(
      const BackendBuffer& dst,
      const BackendBuffer& src,
      const CopyRegion& region,
      const char* op) const {
    if (add_overflows(region.dst_offset, region.size_bytes) ||
        region.dst_offset + region.size_bytes > dst.size_bytes()) {
      return make_error(op, BackendErrorCode::InvalidArgument,
                        "destination copy range exceeds buffer size");
    }
    if (add_overflows(region.src_offset, region.size_bytes) ||
        region.src_offset + region.size_bytes > src.size_bytes()) {
      return make_error(op, BackendErrorCode::InvalidArgument,
                        "source copy range exceeds buffer size");
    }
    return make_success(op);
  }

  BackendError validate_buffer_region(
      const BackendBuffer& buffer,
      const BufferTransferRegion& region,
      const char* op) const {
    if (add_overflows(region.buffer_offset, region.size_bytes) ||
        region.buffer_offset + region.size_bytes > buffer.size_bytes()) {
      return make_error(op, BackendErrorCode::InvalidArgument,
                        "transfer range exceeds buffer size");
    }
    return make_success(op);
  }

  std::mutex m_queues_mutex;
  std::vector<std::weak_ptr<BackendQueue>> m_created_queues;
};

}  // namespace tinygs
