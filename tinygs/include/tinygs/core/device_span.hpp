#pragma once

#include <cstddef>

namespace tinygs {

template <typename T>
class DeviceSpan {
public:
  using value_type = T;

  DeviceSpan() = default;
  DeviceSpan(T* ptr, size_t size) : m_ptr(ptr), m_size(size) {}

  T* data() const { return m_ptr; }
  T* begin() const { return m_ptr; }
  T* end() const { return m_ptr + m_size; }
  size_t size() const { return m_size; }
  T& operator[](size_t idx) const { return m_ptr[idx]; }

private:
  T* m_ptr = nullptr;
  size_t m_size = 0;
};

}  // namespace tinygs
