#pragma once
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/core/image.hpp"
namespace tinygs {

struct Data {
  /// Image Data: for now, the image should have float32 type.
  Image image; // must be stored in cudaMallocHost memory.

  /// Camera Data
  mat4x4 w2c;
  mat3x3 K;
  /// @brief The camera id of the data. (0-based)
  uint32_t cam_uid;

  /// @brief The frame id of the data. (1-based)
  uint32_t frame_uid;
};

class DatasetBase {
public:
  explicit DatasetBase() = default;
  DatasetBase(const DatasetBase&) = delete;
  DatasetBase& operator=(const DatasetBase&) = delete;
  DatasetBase(DatasetBase&&) = default;
  DatasetBase& operator=(DatasetBase&&) = default;

  /// @brief Load the dataset from disk.
  virtual void load() = 0;

  /// @brief Set the parameters of the dataset.
  /// @param params The parameters of the dataset.
  virtual void set_params(const json& params) {}

  /// @brief Get the parameters of the dataset.
  /// @return The parameters of the dataset.
  virtual json get_params() const { return json::object(); }

  virtual SingleCameraLoader& get_camera_loader() = 0;

  virtual ~DatasetBase() = default;

  /// Get the size of the dataset.
  virtual size_t size() const noexcept = 0;

  /// Get the image shape of the dataset.
  /// @return The image shape of the dataset.
  virtual ImageShape image_shape() const = 0;

  /// Get the data at the given index.
  /// @param idx The index of the data.
  /// @return The data at the given index.
  virtual Data operator[](size_t idx) const = 0;
};


/// @brief Create a dataset object.
/// @param dataset_type The type of dataset to create.
/// @return The created dataset.
std::unique_ptr<DatasetBase> create_dataset(const std::string& dataset_type);

} // namespace tinygs