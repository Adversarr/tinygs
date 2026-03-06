#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tinygs/common.hpp"
#include "tinygs/platform/runtime.hpp"

namespace tinygs {

/// @brief Non-owning byte-range view into device memory backed by BackendBuffer.
class BufferView {
public:
  BufferView() = default;

  BufferView(const std::shared_ptr<BackendBuffer>& buffer, size_t offset_bytes, size_t size_bytes)
      : m_buffer(buffer), m_offset_bytes(offset_bytes), m_size_bytes(size_bytes) {}

  const std::shared_ptr<BackendBuffer>& buffer() const { return m_buffer; }
  size_t offset_bytes() const { return m_offset_bytes; }
  size_t size_bytes() const { return m_size_bytes; }
  bool empty() const { return m_size_bytes == 0; }

  const void* data() const {
    if (!m_buffer) return nullptr;
    const auto* bytes = static_cast<const uint8_t*>(m_buffer->data());
    return bytes + m_offset_bytes;
  }

  void* data() {
    return const_cast<void*>(static_cast<const BufferView*>(this)->data());
  }

private:
  std::shared_ptr<BackendBuffer> m_buffer;
  size_t m_offset_bytes = 0;
  size_t m_size_bytes = 0;
};

/// @brief Non-owning typed view over device memory backed by BackendBuffer.
template <typename T>
class DeviceSpan {
public:
  using value_type = T;

  DeviceSpan() = default;

  explicit DeviceSpan(const BufferView& view) : m_view(view) {
    CHECK_THROW(m_view.size_bytes() % sizeof(T) == 0);
  }

  DeviceSpan(const std::shared_ptr<BackendBuffer>& buffer, size_t offset_elems, size_t count_elems)
      : m_view(buffer, offset_elems * sizeof(T), count_elems * sizeof(T)) {}

  explicit DeviceSpan(const std::shared_ptr<BackendBuffer>& buffer)
      : m_view(buffer, 0, buffer ? buffer->size_bytes() : 0) {}

  T* data() const {
    auto& view = const_cast<BufferView&>(m_view);
    return static_cast<T*>(view.data());
  }
  T* begin() const { return data(); }
  T* end() const { return data() + size(); }
  size_t size() const { return m_view.size_bytes() / sizeof(T); }
  T& operator[](size_t idx) const { return data()[idx]; }
  bool empty() const { return m_view.empty(); }

  const BufferView& view() const { return m_view; }

private:
  BufferView m_view;
};

}  // namespace tinygs
