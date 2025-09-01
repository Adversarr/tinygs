#pragma once
#include "tinygs/core/image.hpp"
namespace tinygs {

struct Data {
  /// Image Data: for now, the image should have float32 type.
  Image<const float> image; // must be stored in cudaMallocHost memory.

  /// Camera Data
  mat4x4 w2c;
  mat3x3 K;
};

class DatasetBase {
public:
  explicit DatasetBase() = default;

  virtual ~DatasetBase() = default;

  /// Get the size of the dataset.
  size_t size() const noexcept { return m_size; }

  /// Get the data at the given index.
  /// @param idx The index of the data.
  /// @return The data at the given index.
  virtual Data operator[](size_t idx) const = 0;

protected:
  size_t m_size = 0;
};

} // namespace tinygs