#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinygs/platform/buffer_view.hpp"
#include "tinygs/platform/runtime.hpp"

namespace tinygs {

namespace detail {

inline size_t checked_multiply_size_t(size_t a, size_t b, const char* op_name) {
  if (a == 0 || b == 0) return 0;
  if (a > std::numeric_limits<size_t>::max() / b) {
    throw std::runtime_error(std::string(op_name) + " failed: size overflow");
  }
  return a * b;
}

inline BufferView whole_buffer_view(const std::shared_ptr<BackendBuffer>& buffer) {
  return BufferView(buffer.get(), 0, buffer ? buffer->size_bytes() : 0);
}

inline void throw_if_status_error(const BackendError& status, const char* op_name) {
  if (status.ok()) {
    return;
  }
  throw std::runtime_error(std::string(op_name) + " failed: " + to_string(status));
}

inline void synchronize_queue_or_throw(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const char* op_name) {
  throw_if_status_error(runtime.synchronize_queue(queue), op_name);
}

inline BufferTransferRegion make_transfer_region(
    const BufferView& view,
    size_t size_bytes,
    const char* op_name) {
  if (!view.buffer()) {
    throw std::runtime_error(std::string(op_name) + " failed: buffer view has null buffer");
  }
  if (view.offset_bytes() > view.buffer()->size_bytes()) {
    throw std::runtime_error(std::string(op_name) + " failed: offset out of range");
  }
  if (size_bytes > view.size_bytes()) {
    throw std::runtime_error(std::string(op_name) + " failed: requested size exceeds view");
  }
  if (view.offset_bytes() > std::numeric_limits<size_t>::max() - size_bytes) {
    throw std::runtime_error(std::string(op_name) + " failed: offset overflow");
  }
  const size_t end_offset = view.offset_bytes() + size_bytes;
  if (end_offset > view.buffer()->size_bytes()) {
    throw std::runtime_error(std::string(op_name) + " failed: transfer exceeds buffer bounds");
  }

  BufferTransferRegion region;
  region.size_bytes = size_bytes;
  region.buffer_offset = view.offset_bytes();
  return region;
}

inline CopyRegion make_copy_region(
    const BufferView& dst,
    const BufferView& src,
    size_t size_bytes,
    const char* op_name) {
  if (!dst.buffer() || !src.buffer()) {
    throw std::runtime_error(std::string(op_name) + " failed: buffer view has null buffer");
  }
  if (size_bytes > dst.size_bytes() || size_bytes > src.size_bytes()) {
    throw std::runtime_error(std::string(op_name) + " failed: requested size exceeds view");
  }
  if (dst.offset_bytes() > std::numeric_limits<size_t>::max() - size_bytes ||
      src.offset_bytes() > std::numeric_limits<size_t>::max() - size_bytes) {
    throw std::runtime_error(std::string(op_name) + " failed: offset overflow");
  }
  if (dst.offset_bytes() + size_bytes > dst.buffer()->size_bytes() ||
      src.offset_bytes() + size_bytes > src.buffer()->size_bytes()) {
    throw std::runtime_error(std::string(op_name) + " failed: copy exceeds buffer bounds");
  }

  CopyRegion region;
  region.size_bytes = size_bytes;
  region.dst_offset = dst.offset_bytes();
  region.src_offset = src.offset_bytes();
  return region;
}

inline void validate_raw_copy_args(
    const void* ptr,
    size_t size_bytes,
    const char* arg_name,
    const char* op_name) {
  if (size_bytes > 0 && ptr == nullptr) {
    throw std::runtime_error(std::string(op_name) + " failed: " + arg_name + " must not be null");
  }
}

}  // namespace detail

/// Typed accessor for BackendBuffer data pointer.
/// Interprets the buffer data as `T`.
template <typename T>
T* buffer_data(const std::shared_ptr<BackendBuffer>& buf) {
  if (!buf) return nullptr;
  return static_cast<T*>(buf->data());
}

/// Typed const accessor for BackendBuffer data pointer.
template <typename T>
const T* buffer_data_const(const std::shared_ptr<BackendBuffer>& buf) {
  if (!buf) return nullptr;
  return static_cast<const T*>(buf->data());
}

/// Return the number of elements of type T that fit in the buffer.
template <typename T>
size_t buffer_count(const std::shared_ptr<BackendBuffer>& buf) {
  if (!buf) return 0;
  return buf->size_bytes() / sizeof(T);
}

/// Create a BackendBuffer using the provided descriptor
inline std::shared_ptr<BackendBuffer> create_buffer(
    BackendRuntime& runtime,
    const BufferDesc& desc) {
  auto result = runtime.create_buffer(desc);
  if (!result.ok()) {
    throw std::runtime_error(
        "create_buffer failed (" + desc.debug_name + "): " + to_string(result.error()));
  }
  return result.value();
}

/// Create a device-memory BackendBuffer of the given byte size
inline std::shared_ptr<BackendBuffer> create_device_buffer(
    BackendRuntime& runtime,
    size_t size_bytes,
    const std::string& debug_name = {}) {
  BufferDesc desc;
  desc.size_bytes = size_bytes;
  desc.memory_class = BufferMemoryClass::Device;
  desc.debug_name = debug_name;
  return create_buffer(runtime, desc);
}

/// Create a unified-memory BackendBuffer of the given byte size
inline std::shared_ptr<BackendBuffer> create_unified_buffer(
    BackendRuntime& runtime,
    size_t size_bytes,
    const std::string& debug_name = {},
    BufferHostAccess host_access = BufferHostAccess::ReadWrite) {
  BufferDesc desc;
  desc.size_bytes = size_bytes;
  desc.memory_class = BufferMemoryClass::Unified;
  desc.host_access = host_access;
  desc.debug_name = debug_name;
  return create_buffer(runtime, desc);
}

/// Create a host-pinned BackendBuffer of the given byte size
inline std::shared_ptr<BackendBuffer> create_host_pinned_buffer(
    BackendRuntime& runtime,
    size_t size_bytes,
    const std::string& debug_name = {},
    BufferHostAccess host_access = BufferHostAccess::ReadWrite) {
  BufferDesc desc;
  desc.size_bytes = size_bytes;
  desc.memory_class = BufferMemoryClass::HostPinned;
  desc.host_access = host_access;
  desc.debug_name = debug_name;
  return create_buffer(runtime, desc);
}

/// Create a device-memory BackendBuffer sized for `count` elements of type T.
template <typename T>
std::shared_ptr<BackendBuffer> create_device_buffer_for(
    BackendRuntime& runtime,
    size_t count,
    const std::string& debug_name = {}) {
  return create_device_buffer(
      runtime,
      detail::checked_multiply_size_t(count, sizeof(T), "create_device_buffer_for"),
      debug_name);
}

/// Create a unified-memory BackendBuffer sized for `count` elements of type T.
template <typename T>
std::shared_ptr<BackendBuffer> create_unified_buffer_for(
    BackendRuntime& runtime,
    size_t count,
    const std::string& debug_name = {},
    BufferHostAccess host_access = BufferHostAccess::ReadWrite) {
  return create_unified_buffer(
      runtime,
      detail::checked_multiply_size_t(count, sizeof(T), "create_unified_buffer_for"),
      debug_name,
      host_access);
}

/// Create a host-pinned BackendBuffer sized for `count` elements of type T.
template <typename T>
std::shared_ptr<BackendBuffer> create_host_pinned_buffer_for(
    BackendRuntime& runtime,
    size_t count,
    const std::string& debug_name = {},
    BufferHostAccess host_access = BufferHostAccess::ReadWrite) {
  return create_host_pinned_buffer(
      runtime,
      detail::checked_multiply_size_t(count, sizeof(T), "create_host_pinned_buffer_for"),
      debug_name,
      host_access);
}

/// Fill a buffer asynchronously via the runtime
inline void fill_buffer_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    uint8_t value) {
  if (!buffer) return;
  detail::throw_if_status_error(runtime.fill_buffer_async(queue, *buffer, value), "fill_buffer_async");
}

/// Fill a buffer view asynchronously via the runtime
inline void fill_buffer_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& view,
    uint8_t value) {
  if (view.empty()) return;
  detail::throw_if_status_error(
      runtime.fill_buffer_async(queue, *view.buffer(), value, view.offset_bytes(), view.size_bytes()),
      "fill_buffer_async");
}

/// Fill a buffer synchronously via the runtime
inline void fill_buffer(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    uint8_t value) {
  fill_buffer_async(runtime, queue, buffer, value);
  detail::synchronize_queue_or_throw(runtime, queue, "fill_buffer sync");
}

/// Fill a buffer view synchronously via the runtime
inline void fill_buffer(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& view,
    uint8_t value) {
  fill_buffer_async(runtime, queue, view, value);
  detail::synchronize_queue_or_throw(runtime, queue, "fill_buffer sync");
}

/// Alias for zero-filling a buffer asynchronously.
inline void fill_buffer_zero_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer) {
  fill_buffer_async(runtime, queue, buffer, 0);
}

/// Alias for zero-filling a buffer view asynchronously.
inline void fill_buffer_zero_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& view) {
  fill_buffer_async(runtime, queue, view, 0);
}

/// Alias for zero-filling a buffer synchronously.
inline void fill_buffer_zero(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer) {
  fill_buffer(runtime, queue, buffer, 0);
}

/// Alias for zero-filling a buffer view synchronously.
inline void fill_buffer_zero(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& view) {
  fill_buffer(runtime, queue, view, 0);
}

/// Copy data from host to device buffer view asynchronously
template <typename T>
void copy_from_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const T* src,
    size_t count) {
  const size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_from_host_async");
  if (size_bytes > 0 && src == nullptr) {
    throw std::runtime_error("copy_from_host_async failed: src must not be null");
  }
  const auto region = detail::make_transfer_region(dst, size_bytes, "copy_from_host_async");
  detail::throw_if_status_error(runtime.copy_from_host_async(queue, *dst.buffer(), src, region), "copy_from_host_async");
}

/// Copy data from host to device buffer view synchronously
template <typename T>
void copy_from_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const T* src,
    size_t count) {
  copy_from_host_async(runtime, queue, dst, src, count);
  detail::synchronize_queue_or_throw(runtime, queue, "copy_from_host sync");
}

/// Copy data from host vector to device buffer view synchronously
template <typename T>
void copy_from_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const std::vector<T>& src) {
  copy_from_host(runtime, queue, dst, src.data(), src.size());
}

/// Copy data from host vector to device buffer view asynchronously
template <typename T>
void copy_from_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const std::vector<T>& src) {
  copy_from_host_async(runtime, queue, dst, src.data(), src.size());
}

/// Copy data from host to device buffer asynchronously
template <typename T>
void copy_from_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count) {
  copy_from_host_async(runtime, queue, detail::whole_buffer_view(buffer), src, count);
}

/// Copy data from host to device buffer synchronously
template <typename T>
void copy_from_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count) {
  copy_from_host(runtime, queue, detail::whole_buffer_view(buffer), src, count);
}

/// Copy data from host vector to device buffer asynchronously
template <typename T>
void copy_from_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src) {
  copy_from_host_async(runtime, queue, buffer, src.data(), src.size());
}

/// Copy data from host vector to device buffer synchronously
template <typename T>
void copy_from_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src) {
  copy_from_host(runtime, queue, buffer, src.data(), src.size());
}

/// Copy data from host to device buffer synchronously with an element offset.
template <typename T>
void copy_from_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count,
    size_t buffer_offset_elems) {
  const size_t offset_bytes = detail::checked_multiply_size_t(buffer_offset_elems, sizeof(T), "copy_from_host");
  const size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_from_host");
  copy_from_host(runtime, queue, BufferView(buffer.get(), offset_bytes, size_bytes), src, count);
}

/// Copy data from host to device buffer asynchronously with an element offset.
template <typename T>
void copy_from_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count,
    size_t buffer_offset_elems) {
  const size_t offset_bytes = detail::checked_multiply_size_t(buffer_offset_elems, sizeof(T), "copy_from_host_async");
  const size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_from_host_async");
  copy_from_host_async(runtime, queue, BufferView(buffer.get(), offset_bytes, size_bytes), src, count);
}

template <typename T>
void copy_from_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src,
    size_t buffer_offset_elems) {
  copy_from_host(runtime, queue, buffer, src.data(), src.size(), buffer_offset_elems);
}

template <typename T>
void copy_from_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src,
    size_t buffer_offset_elems) {
  copy_from_host_async(runtime, queue, buffer, src.data(), src.size(), buffer_offset_elems);
}

/// Copy data from device buffer view to host asynchronously
template <typename T>
void copy_to_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& src,
    T* dst,
    size_t count) {
  const size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_to_host_async");
  if (size_bytes > 0 && dst == nullptr) {
    throw std::runtime_error("copy_to_host_async failed: dst must not be null");
  }
  const auto region = detail::make_transfer_region(src, size_bytes, "copy_to_host_async");
  detail::throw_if_status_error(runtime.copy_to_host_async(queue, dst, *src.buffer(), region), "copy_to_host_async");
}

/// Copy data from device buffer view to host synchronously
template <typename T>
void copy_to_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& src,
    T* dst,
    size_t count) {
  copy_to_host_async(runtime, queue, src, dst, count);
  detail::synchronize_queue_or_throw(runtime, queue, "copy_to_host sync");
}

/// Copy data from device buffer to host synchronously
template <typename T>
void copy_to_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    T* dst,
    size_t count) {
  copy_to_host(runtime, queue, detail::whole_buffer_view(buffer), dst, count);
}

/// Copy data from device buffer to host asynchronously
template <typename T>
void copy_to_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    T* dst,
    size_t count) {
  copy_to_host_async(runtime, queue, detail::whole_buffer_view(buffer), dst, count);
}

/// Copy data from device buffer view to host vector. Resizes dst if needed.
template <typename T>
void copy_to_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& src,
    std::vector<T>& dst) {
  const size_t count = src.size_bytes() / sizeof(T);
  if (dst.size() < count) {
    dst.resize(count);
  }
  copy_to_host(runtime, queue, src, dst.data(), count);
}

/// Copy data from device buffer view to host vector asynchronously. Resizes dst if needed.
template <typename T>
void copy_to_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& src,
    std::vector<T>& dst) {
  const size_t count = src.size_bytes() / sizeof(T);
  if (dst.size() < count) {
    dst.resize(count);
  }
  copy_to_host_async(runtime, queue, src, dst.data(), count);
}

/// Copy data from device buffer to host vector asynchronously. Resizes dst if needed.
template <typename T>
void copy_to_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    std::vector<T>& dst) {
  const size_t count = buffer_count<T>(buffer);
  if (dst.size() < count) {
    dst.resize(count);
  }
  copy_to_host_async(runtime, queue, buffer, dst.data(), count);
}

/// Copy data from device buffer to host vector. Resizes dst if needed.
template <typename T>
void copy_to_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    std::vector<T>& dst) {
  const size_t count = buffer_count<T>(buffer);
  if (dst.size() < count) {
    dst.resize(count);
  }
  copy_to_host(runtime, queue, buffer, dst.data(), count);
}

/// Copy data from host memory to a raw device pointer asynchronously
inline void copy_raw_host_to_device_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  detail::validate_raw_copy_args(dst, size_bytes, "dst", "copy_raw_host_to_device_async");
  detail::validate_raw_copy_args(src, size_bytes, "src", "copy_raw_host_to_device_async");
  detail::throw_if_status_error(
      runtime.copy_host_to_device_async(queue, dst, src, size_bytes),
      "copy_raw_host_to_device_async");
}

/// Copy data from host memory to a raw device pointer synchronously
inline void copy_raw_host_to_device(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  copy_raw_host_to_device_async(runtime, queue, dst, src, size_bytes);
  detail::synchronize_queue_or_throw(runtime, queue, "copy_raw_host_to_device sync");
}

/// Copy data from raw device memory to host asynchronously
inline void copy_raw_device_to_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  detail::validate_raw_copy_args(dst, size_bytes, "dst", "copy_raw_device_to_host_async");
  detail::validate_raw_copy_args(src, size_bytes, "src", "copy_raw_device_to_host_async");
  detail::throw_if_status_error(
      runtime.copy_device_to_host_async(queue, dst, src, size_bytes),
      "copy_raw_device_to_host_async");
}

/// Copy data from raw device memory to host synchronously
inline void copy_raw_device_to_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  copy_raw_device_to_host_async(runtime, queue, dst, src, size_bytes);
  detail::synchronize_queue_or_throw(runtime, queue, "copy_raw_device_to_host sync");
}

/// Copy data from a buffer view to host asynchronously using raw host memory.
inline void copy_raw_device_to_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const BufferView& src) {
  copy_raw_device_to_host_async(runtime, queue, dst, src.data(), src.size_bytes());
}

/// Copy data from a buffer view to host synchronously using raw host memory.
inline void copy_raw_device_to_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const BufferView& src) {
  copy_raw_device_to_host(runtime, queue, dst, src.data(), src.size_bytes());
}

/// Alias for raw host-to-device copies.
inline void copy_host_to_device_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  copy_raw_host_to_device_async(runtime, queue, dst, src, size_bytes);
}

/// Alias for raw host-to-device copies.
inline void copy_host_to_device(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  copy_raw_host_to_device(runtime, queue, dst, src, size_bytes);
}

/// Alias for raw device-to-host copies.
inline void copy_device_to_host_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const BufferView& src) {
  copy_raw_device_to_host_async(runtime, queue, dst, src);
}

/// Alias for raw device-to-host copies.
inline void copy_device_to_host(
    BackendRuntime& runtime,
    BackendQueue& queue,
    void* dst,
    const BufferView& src) {
  copy_raw_device_to_host(runtime, queue, dst, src);
}

/// Copy data between device buffer views asynchronously
inline void copy_buffer_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const BufferView& src,
    size_t size_bytes) {
  const auto region = detail::make_copy_region(dst, src, size_bytes, "copy_buffer_async");
  detail::throw_if_status_error(runtime.copy_buffer_async(queue, *dst.buffer(), *src.buffer(), region), "copy_buffer_async");
}

/// Copy full source view to destination view asynchronously
inline void copy_buffer_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const BufferView& src) {
  copy_buffer_async(runtime, queue, dst, src, src.size_bytes());
}

/// Copy data between device buffer views synchronously
inline void copy_buffer(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const BufferView& src,
    size_t size_bytes) {
  copy_buffer_async(runtime, queue, dst, src, size_bytes);
  detail::synchronize_queue_or_throw(runtime, queue, "copy_buffer sync");
}

/// Copy full source view to destination view synchronously
inline void copy_buffer(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const BufferView& dst,
    const BufferView& src) {
  copy_buffer(runtime, queue, dst, src, src.size_bytes());
}

/// Copy data between device buffers asynchronously
inline void copy_buffer_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& dst,
    const std::shared_ptr<BackendBuffer>& src,
    size_t size_bytes,
    size_t dst_offset = 0,
    size_t src_offset = 0) {
  copy_buffer_async(
      runtime,
      queue,
      BufferView(dst.get(), dst_offset, size_bytes),
      BufferView(src.get(), src_offset, size_bytes),
      size_bytes);
}

/// Copy data between device buffers synchronously
inline void copy_buffer(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& dst,
    const std::shared_ptr<BackendBuffer>& src,
    size_t size_bytes,
    size_t dst_offset = 0,
    size_t src_offset = 0) {
  copy_buffer_async(runtime, queue, dst, src, size_bytes, dst_offset, src_offset);
  detail::synchronize_queue_or_throw(runtime, queue, "copy_buffer sync");
}

/// Clone a device buffer (create a copy with same contents)
inline std::shared_ptr<BackendBuffer> clone_buffer(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& src,
    const std::string& debug_name = {}) {
  if (!src) return nullptr;
  auto dst = create_device_buffer(runtime, src->size_bytes(), debug_name);
  if (src->size_bytes() > 0) {
    copy_buffer(runtime, queue, dst, src, src->size_bytes());
  }
  return dst;
}

/// Clone a device buffer asynchronously
inline std::shared_ptr<BackendBuffer> clone_buffer_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& src,
    const std::string& debug_name = {}) {
  if (!src) return nullptr;
  auto dst = create_device_buffer(runtime, src->size_bytes(), debug_name);
  if (src->size_bytes() > 0) {
    copy_buffer_async(runtime, queue, dst, src, src->size_bytes());
  }
  return dst;
}

/// Resize a buffer: create new buffer, copy old data, fill new region.
template <typename T>
std::shared_ptr<BackendBuffer> resize_buffer(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& old_buf,
    size_t new_count,
    const std::string& debug_name = {}) {
  if (new_count == 0) return nullptr;

  const size_t new_size = detail::checked_multiply_size_t(new_count, sizeof(T), "resize_buffer");
  auto new_buf = create_device_buffer(runtime, new_size, debug_name);
  fill_buffer(runtime, queue, new_buf, 0);

  if (old_buf && old_buf->size_bytes() > 0) {
    const size_t copy_size = std::min(old_buf->size_bytes(), new_size);
    copy_buffer(runtime, queue, new_buf, old_buf, copy_size);
  }

  return new_buf;
}

/// Resize a buffer asynchronously.
template <typename T>
std::shared_ptr<BackendBuffer> resize_buffer_async(
    BackendRuntime& runtime,
    BackendQueue& queue,
    const std::shared_ptr<BackendBuffer>& old_buf,
    size_t new_count,
    const std::string& debug_name = {}) {
  if (new_count == 0) return nullptr;

  const size_t new_size = detail::checked_multiply_size_t(new_count, sizeof(T), "resize_buffer_async");
  auto new_buf = create_device_buffer(runtime, new_size, debug_name);
  fill_buffer_async(runtime, queue, new_buf, 0);

  if (old_buf && old_buf->size_bytes() > 0) {
    const size_t copy_size = std::min(old_buf->size_bytes(), new_size);
    copy_buffer_async(runtime, queue, new_buf, old_buf, copy_size);
  }

  return new_buf;
}

}  // namespace tinygs
