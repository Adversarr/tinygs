#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tinygs/platform/runtime_contract.hpp"

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

}  // namespace tinygs
