#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include "tinygs/platform/runtime_contract.hpp"

namespace tinygs {

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
  auto status = runtime->fill_buffer_async(queue, buffer, 0);
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
  BufferTransferRegion region;
  region.size_bytes = count * sizeof(T);
  region.buffer_offset = 0;
  auto status = runtime->copy_from_host_async(queue, buffer, src, region);
  if (!status.ok()) {
    throw std::runtime_error("copy_from_host failed: " + to_string(status));
  }
  auto sync_status = runtime->synchronize_queue(queue);
  if (!sync_status.ok()) {
    throw std::runtime_error("copy_from_host sync failed: " + to_string(sync_status));
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
  BufferTransferRegion region;
  region.size_bytes = count * sizeof(T);
  region.buffer_offset = 0;
  auto status = runtime->copy_to_host_async(queue, dst, buffer, region);
  if (!status.ok()) {
    throw std::runtime_error("copy_to_host failed: " + to_string(status));
  }
  auto sync_status = runtime->synchronize_queue(queue);
  if (!sync_status.ok()) {
    throw std::runtime_error("copy_to_host sync failed: " + to_string(sync_status));
  }
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

}  // namespace tinygs
