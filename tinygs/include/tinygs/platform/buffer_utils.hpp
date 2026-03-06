#pragma once

#include <algorithm>
#include <cstddef>
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
  return BufferView(buffer, 0, buffer ? buffer->size_bytes() : 0);
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
  size_t end_offset = view.offset_bytes() + size_bytes;
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

}  // namespace detail

/// @brief Typed accessor for BackendBuffer data pointer.
/// @tparam T Element type to interpret the buffer data as.
template <typename T>
T* buffer_data(const std::shared_ptr<BackendBuffer>& buf) {
  if (!buf) return nullptr;
  return static_cast<T*>(buf->data());
}

/// @brief Typed const accessor for BackendBuffer data pointer.
template <typename T>
const T* buffer_data_const(const std::shared_ptr<BackendBuffer>& buf) {
  if (!buf) return nullptr;
  return static_cast<const T*>(buf->data());
}

/// @brief Return the number of elements of type T that fit in the buffer.
template <typename T>
size_t buffer_count(const std::shared_ptr<BackendBuffer>& buf) {
  if (!buf) return 0;
  return buf->size_bytes() / sizeof(T);
}

/// @brief Create a device-memory BackendBuffer of the given byte size.
///        Throws on failure.
inline std::shared_ptr<BackendBuffer> create_device_buffer(
    const std::shared_ptr<BackendRuntime>& runtime,
    size_t size_bytes,
    const std::string& debug_name = {}) {
  BufferDesc desc;
  desc.size_bytes = size_bytes;
  desc.memory_class = BufferMemoryClass::Device;
  desc.debug_name = debug_name;
  auto result = runtime->create_buffer(desc);
  if (!result.ok()) {
    throw std::runtime_error(
        "create_device_buffer failed (" + debug_name + "): " + to_string(result.error()));
  }
  return result.value();
}

/// @brief Create a device-memory BackendBuffer sized for `count` elements of type T.
///        Throws on failure.
template <typename T>
std::shared_ptr<BackendBuffer> create_device_buffer_for(
    const std::shared_ptr<BackendRuntime>& runtime,
    size_t count,
    const std::string& debug_name = {}) {
  return create_device_buffer(runtime, count * sizeof(T), debug_name);
}

/// @brief Helper: fill buffer with zeros asynchronously via the runtime.
///        Throws on failure.
inline void fill_buffer_zero(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer) {
  if (!buffer) return;
  auto status = runtime->fill_buffer_async(queue, buffer, 0, 0, buffer->size_bytes());
  if (!status.ok()) {
    throw std::runtime_error("fill_buffer_zero failed: " + to_string(status));
  }
}

/// @brief Copy data from host to device buffer. Throws on failure.
template <typename T>
void copy_from_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count) {
  copy_from_host(runtime, queue, detail::whole_buffer_view(buffer), src, count);
}

/// @brief Copy data from host to device buffer view asynchronously. Throws on failure.
template <typename T>
void copy_from_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const T* src,
    size_t count) {
  size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_from_host_async");
  if (size_bytes > 0 && src == nullptr) {
    throw std::runtime_error("copy_from_host_async failed: src must not be null");
  }
  auto region = detail::make_transfer_region(dst, size_bytes, "copy_from_host_async");
  auto status = runtime->copy_from_host_async(queue, dst.buffer(), src, region);
  if (!status.ok()) {
    throw std::runtime_error("copy_from_host_async failed: " + to_string(status));
  }
}

/// @brief Copy data from host to device buffer view. Throws on failure.
template <typename T>
void copy_from_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const T* src,
    size_t count) {
  copy_from_host_async(runtime, queue, dst, src, count);
  auto sync_status = runtime->synchronize_queue(queue);
  if (!sync_status.ok()) {
    throw std::runtime_error("copy_from_host sync failed: " + to_string(sync_status));
  }
}

/// @brief Copy data from host vector to device buffer view. Throws on failure.
template <typename T>
void copy_from_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const std::vector<T>& src) {
  copy_from_host(runtime, queue, dst, src.data(), src.size());
}

/// @brief Copy data from host vector to device buffer view asynchronously. Throws on failure.
template <typename T>
void copy_from_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const std::vector<T>& src) {
  copy_from_host_async(runtime, queue, dst, src.data(), src.size());
}

/// @brief Copy data from host to device buffer asynchronously. Throws on failure.
template <typename T>
void copy_from_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count) {
  copy_from_host_async(runtime, queue, detail::whole_buffer_view(buffer), src, count);
}

/// @brief Copy data from host vector to device buffer asynchronously. Throws on failure.
template <typename T>
void copy_from_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src) {
  copy_from_host_async(runtime, queue, buffer, src.data(), src.size());
}

/// @brief Copy data from host to device buffer. Throws on failure.
template <typename T>
void copy_from_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count,
    size_t buffer_offset_elems) {
  size_t offset_bytes = detail::checked_multiply_size_t(buffer_offset_elems, sizeof(T), "copy_from_host");
  size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_from_host");
  copy_from_host(runtime, queue, BufferView(buffer, offset_bytes, size_bytes), src, count);
}

/// @brief Copy data from host to device buffer asynchronously. Throws on failure.
template <typename T>
void copy_from_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const T* src,
    size_t count,
    size_t buffer_offset_elems) {
  size_t offset_bytes = detail::checked_multiply_size_t(buffer_offset_elems, sizeof(T), "copy_from_host_async");
  size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_from_host_async");
  copy_from_host_async(runtime, queue, BufferView(buffer, offset_bytes, size_bytes), src, count);
}

template <typename T>
void copy_from_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src,
    size_t buffer_offset_elems) {
  copy_from_host(runtime, queue, buffer, src.data(), src.size(), buffer_offset_elems);
}

template <typename T>
void copy_from_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src,
    size_t buffer_offset_elems) {
  copy_from_host_async(runtime, queue, buffer, src.data(), src.size(), buffer_offset_elems);
}

/// @brief Copy data from device buffer to host asynchronously. Throws on failure.
template <typename T>
void copy_to_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& src,
    T* dst,
    size_t count) {
  size_t size_bytes = detail::checked_multiply_size_t(count, sizeof(T), "copy_to_host_async");
  if (size_bytes > 0 && dst == nullptr) {
    throw std::runtime_error("copy_to_host_async failed: dst must not be null");
  }
  auto region = detail::make_transfer_region(src, size_bytes, "copy_to_host_async");
  auto status = runtime->copy_to_host_async(queue, dst, src.buffer(), region);
  if (!status.ok()) {
    throw std::runtime_error("copy_to_host_async failed: " + to_string(status));
  }
}

/// @brief Copy data from device buffer to host. Throws on failure.
template <typename T>
void copy_to_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& src,
    T* dst,
    size_t count) {
  copy_to_host_async(runtime, queue, src, dst, count);
  auto sync_status = runtime->synchronize_queue(queue);
  if (!sync_status.ok()) {
    throw std::runtime_error("copy_to_host sync failed: " + to_string(sync_status));
  }
}

/// @brief Copy data from host vector to device buffer. Throws on failure.
template <typename T>
void copy_from_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    const std::vector<T>& src) {
  copy_from_host(runtime, queue, buffer, src.data(), src.size());
}

/// @brief Copy data from device buffer to host. Throws on failure.
template <typename T>
void copy_to_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    T* dst,
    size_t count) {
  copy_to_host(runtime, queue, detail::whole_buffer_view(buffer), dst, count);
}

/// @brief Copy data from device buffer to host asynchronously. Throws on failure.
template <typename T>
void copy_to_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    T* dst,
    size_t count) {
  copy_to_host_async(runtime, queue, detail::whole_buffer_view(buffer), dst, count);
}

/// @brief Copy data from device buffer view to host vector. Resizes dst if needed. Throws on failure.
template <typename T>
void copy_to_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& src,
    std::vector<T>& dst) {
  size_t count = src.size_bytes() / sizeof(T);
  if (dst.size() < count) {
    dst.resize(count);
  }
  copy_to_host(runtime, queue, src, dst.data(), count);
}

/// @brief Copy data from device buffer view to host vector asynchronously. Resizes dst if needed. Throws on failure.
template <typename T>
void copy_to_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& src,
    std::vector<T>& dst) {
  size_t count = src.size_bytes() / sizeof(T);
  if (dst.size() < count) {
    dst.resize(count);
  }
  copy_to_host_async(runtime, queue, src, dst.data(), count);
}

/// @brief Copy data from device buffer to host vector. Resizes dst if needed. Throws on failure.
template <typename T>
void copy_to_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buffer,
    std::vector<T>& dst) {
  size_t count = buffer_count<T>(buffer);
  if (dst.size() < count) {
    dst.resize(count);
  }
  copy_to_host(runtime, queue, buffer, dst.data(), count);
}

/// @brief Copy data from device memory to host asynchronously using a raw device view.
inline void copy_device_to_host_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    void* dst,
    const BufferView& src) {
  if (src.size_bytes() > 0 && dst == nullptr) {
    throw std::runtime_error("copy_device_to_host_async failed: dst must not be null");
  }
  if (src.size_bytes() == 0) return;
  if (!src.data()) {
    throw std::runtime_error("copy_device_to_host_async failed: src must not be null");
  }
  auto status = runtime->copy_device_to_host_async(queue, dst, src.data(), src.size_bytes());
  if (!status.ok()) {
    throw std::runtime_error("copy_device_to_host_async failed: " + to_string(status));
  }
}

/// @brief Copy data from device memory to host using a raw device view.
inline void copy_device_to_host(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    void* dst,
    const BufferView& src) {
  copy_device_to_host_async(runtime, queue, dst, src);
  auto status = runtime->synchronize_queue(queue);
  if (!status.ok()) {
    throw std::runtime_error("copy_device_to_host sync failed: " + to_string(status));
  }
}

/// @brief Copy data from host memory to a raw device pointer asynchronously. Throws on failure.
inline void copy_host_to_device_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  if (size_bytes > 0 && dst == nullptr) {
    throw std::runtime_error("copy_host_to_device_async failed: dst must not be null");
  }
  if (size_bytes > 0 && src == nullptr) {
    throw std::runtime_error("copy_host_to_device_async failed: src must not be null");
  }
  auto status = runtime->copy_host_to_device_async(queue, dst, src, size_bytes);
  if (!status.ok()) {
    throw std::runtime_error("copy_host_to_device_async failed: " + to_string(status));
  }
}

/// @brief Copy data from host memory to a raw device pointer synchronously. Throws on failure.
inline void copy_host_to_device(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    void* dst,
    const void* src,
    size_t size_bytes) {
  copy_host_to_device_async(runtime, queue, dst, src, size_bytes);
  auto status = runtime->synchronize_queue(queue);
  if (!status.ok()) {
    throw std::runtime_error("copy_host_to_device sync failed: " + to_string(status));
  }
}

/// @brief Copy data between device buffer views asynchronously. Throws on failure.
inline void copy_buffer_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const BufferView& src,
    size_t size_bytes) {
  auto region = detail::make_copy_region(dst, src, size_bytes, "copy_buffer_async");
  auto status = runtime->copy_buffer_async(queue, dst.buffer(), src.buffer(), region);
  if (!status.ok()) {
    throw std::runtime_error("copy_buffer_async failed: " + to_string(status));
  }
}

/// @brief Copy full source view to destination view asynchronously. Throws on failure.
inline void copy_buffer_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const BufferView& src) {
  copy_buffer_async(runtime, queue, dst, src, src.size_bytes());
}

/// @brief Copy data between device buffer views synchronously. Throws on failure.
inline void copy_buffer(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const BufferView& src,
    size_t size_bytes) {
  copy_buffer_async(runtime, queue, dst, src, size_bytes);
  auto status = runtime->synchronize_queue(queue);
  if (!status.ok()) {
    throw std::runtime_error("copy_buffer sync failed: " + to_string(status));
  }
}

/// @brief Copy full source view to destination view synchronously. Throws on failure.
inline void copy_buffer(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const BufferView& dst,
    const BufferView& src) {
  copy_buffer(runtime, queue, dst, src, src.size_bytes());
}

/// @brief Copy data between device buffers asynchronously. Throws on failure.
inline void copy_buffer_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& dst,
    const std::shared_ptr<BackendBuffer>& src,
    size_t size_bytes,
    size_t dst_offset = 0,
    size_t src_offset = 0) {
  copy_buffer_async(
      runtime,
      queue,
      BufferView(dst, dst_offset, size_bytes),
      BufferView(src, src_offset, size_bytes),
      size_bytes);
}

/// @brief Copy data between device buffers synchronously. Throws on failure.
inline void copy_buffer(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& dst,
    const std::shared_ptr<BackendBuffer>& src,
    size_t size_bytes,
    size_t dst_offset = 0,
    size_t src_offset = 0) {
  copy_buffer_async(runtime, queue, dst, src, size_bytes, dst_offset, src_offset);
  auto status = runtime->synchronize_queue(queue);
  if (!status.ok()) {
    throw std::runtime_error("copy_buffer sync failed: " + to_string(status));
  }
}

/// @brief Clone a device buffer (create a copy with same contents). Throws on failure.
inline std::shared_ptr<BackendBuffer> clone_buffer(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& src,
    const std::string& debug_name = {}) {
  if (!src) return nullptr;
  auto dst = create_device_buffer(runtime, src->size_bytes(), debug_name);
  if (src->size_bytes() > 0) {
    copy_buffer(runtime, queue, dst, src, src->size_bytes());
  }
  return dst;
}

/// @brief Clone a device buffer asynchronously. Throws on failure.
inline std::shared_ptr<BackendBuffer> clone_buffer_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& src,
    const std::string& debug_name = {}) {
  if (!src) return nullptr;
  auto dst = create_device_buffer(runtime, src->size_bytes(), debug_name);
  if (src->size_bytes() > 0) {
    copy_buffer_async(runtime, queue, dst, src, src->size_bytes());
  }
  return dst;
}

/// @brief Resize a buffer: create new buffer, copy old data, fill new region with zeros.
///        Returns new buffer. If new_count == 0, returns nullptr.
///        Throws on failure.
template <typename T>
std::shared_ptr<BackendBuffer> resize_buffer(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& old_buf,
    size_t new_count,
    const std::string& debug_name = {}) {
  if (new_count == 0) return nullptr;
  
  size_t new_size = new_count * sizeof(T);
  auto new_buf = create_device_buffer(runtime, new_size, debug_name);
  
  fill_buffer_zero(runtime, queue, new_buf);
  
  if (old_buf && old_buf->size_bytes() > 0) {
    size_t copy_size = std::min(old_buf->size_bytes(), new_size);
    copy_buffer(runtime, queue, new_buf, old_buf, copy_size);
  }
  
  return new_buf;
}

/// @brief Resize a buffer asynchronously.
///        Note: Caller must synchronize queue before using the buffer.
template <typename T>
std::shared_ptr<BackendBuffer> resize_buffer_async(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& old_buf,
    size_t new_count,
    const std::string& debug_name = {}) {
  if (new_count == 0) return nullptr;
  
  size_t new_size = new_count * sizeof(T);
  auto new_buf = create_device_buffer(runtime, new_size, debug_name);
  
  fill_buffer_zero(runtime, queue, new_buf);
  
  if (old_buf && old_buf->size_bytes() > 0) {
    size_t copy_size = std::min(old_buf->size_bytes(), new_size);
    copy_buffer_async(runtime, queue, new_buf, old_buf, copy_size);
  }
  
  return new_buf;
}

}  // namespace tinygs
