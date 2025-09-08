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
  DatasetBase(const DatasetBase&) = delete;
  DatasetBase& operator=(const DatasetBase&) = delete;
  DatasetBase(DatasetBase&&) = default;
  DatasetBase& operator=(DatasetBase&&) = default;

  virtual ~DatasetBase() = default;

  /// Get the size of the dataset.
  virtual size_t size() const noexcept = 0;

  /// Get the image shape of the dataset.
  virtual ImageShape image_shape() const = 0;

  /// Get the data at the given index.
  /// @param idx The index of the data.
  /// @return The data at the given index.
  virtual Data operator[](size_t idx) const = 0;

  // /// Apply a transform to the dataset. (e.g. camera poses.)
  // /// @param transform The transform to apply.
  // virtual void apply_transform(const mat4x4& transform) = 0;

  // TODO: Pose Optimizer.
};

} // namespace tinygs