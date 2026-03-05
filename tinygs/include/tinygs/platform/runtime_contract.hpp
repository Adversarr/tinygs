#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "tinygs/platform/backend_error.hpp"

namespace tinygs {

template <typename T>
class Result {
public:
  Result() = default;
  explicit Result(BackendError error) : m_error(std::move(error)) {}
  Result(std::shared_ptr<T> value, BackendError error)
      : m_value(std::move(value)), m_error(std::move(error)) {}

  static Result<T> success(std::shared_ptr<T> value,
                            BackendType backend,
                            std::string operation = {}) {
    return Result<T>(std::move(value),
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

struct QueueDesc {
  bool non_blocking = true;
  std::string debug_name;
};

struct EventDesc {
  bool disable_timing = true;
  std::string debug_name;
};

enum class BufferMemoryClass : uint8_t {
  Device = 0,
  Unified = 1,
  HostPinned = 2,
};

struct BufferDesc {
  size_t size_bytes = 0;
  size_t alignment = 256;
  BufferMemoryClass memory_class = BufferMemoryClass::Device;
  bool host_visible = false;
  bool interop = false;
  std::string debug_name;
};

class BackendQueue {
public:
  virtual ~BackendQueue() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual void* native_handle() const noexcept = 0;
};

class BackendEvent {
public:
  virtual ~BackendEvent() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual void* native_handle() const noexcept = 0;
};

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

class BackendRuntime {
public:
  virtual ~BackendRuntime() = default;

  virtual BackendType backend_type() const noexcept = 0;
  virtual int device() const noexcept = 0;
  virtual CapabilityProfile capability_profile() const = 0;

  virtual Result<BackendQueue> create_queue(const QueueDesc& desc) = 0;
  virtual Result<BackendEvent> create_event(const EventDesc& desc) = 0;
  virtual Result<BackendBuffer> create_buffer(const BufferDesc& desc) = 0;

  virtual BackendError record_event(const std::shared_ptr<BackendQueue>& queue,
                                    const std::shared_ptr<BackendEvent>& event) = 0;
  virtual BackendError wait_event(const std::shared_ptr<BackendQueue>& queue,
                                  const std::shared_ptr<BackendEvent>& event) = 0;

  virtual BackendError synchronize_queue(const std::shared_ptr<BackendQueue>& queue) = 0;
  virtual BackendError synchronize_event(const std::shared_ptr<BackendEvent>& event) = 0;
  virtual BackendError synchronize_device() = 0;

  virtual BackendError copy_buffer_async(const std::shared_ptr<BackendQueue>& queue,
                                         const std::shared_ptr<BackendBuffer>& dst,
                                         const std::shared_ptr<BackendBuffer>& src,
                                         size_t size_bytes,
                                         size_t dst_offset = 0,
                                         size_t src_offset = 0) = 0;
  virtual BackendError copy_from_host_async(const std::shared_ptr<BackendQueue>& queue,
                                            const std::shared_ptr<BackendBuffer>& dst,
                                            const void* src,
                                            size_t size_bytes,
                                            size_t dst_offset = 0) = 0;
  virtual BackendError copy_to_host_async(const std::shared_ptr<BackendQueue>& queue,
                                          void* dst,
                                          const std::shared_ptr<BackendBuffer>& src,
                                          size_t size_bytes,
                                          size_t src_offset = 0) = 0;
  virtual BackendError copy_device_to_host_async(const std::shared_ptr<BackendQueue>& queue,
                                                 void* dst,
                                                 const void* src,
                                                 size_t size_bytes) = 0;
};

}  // namespace tinygs
