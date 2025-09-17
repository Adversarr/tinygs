#pragma once
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/core/image.hpp"
namespace tinygs {

struct Data {
  Image image;    ///< Image data (float32, cudaMallocHost memory)

  /// Camera data
  mat4x4 w2c;
  mat3x3 K;
  uuid_t cam_uid;    ///< Camera ID (0-based)
  uuid_t frame_idx;  ///< Frame ID (1-based)
  uuid_t timestamp;  ///< Timestamp ID
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

  /// @brief Set dataset parameters
  virtual void set_params(const json& params) {}

  /// @brief Get dataset parameters
  virtual json get_params() const { return json::object(); }

  virtual ~DatasetBase() = default;

  /// @brief Get the size of the dataset.
  virtual size_t size() const noexcept = 0;

  /// @brief Get image shape
  virtual ImageShape image_shape() const = 0;

  /// @brief Get data at index
  virtual Data operator[](size_t idx) const = 0;


  SingleCameraLoader& get_camera_loader() { return m_camera_loader; }
  const SingleCameraLoader& get_camera_loader() const { return m_camera_loader; }

protected:
  SingleCameraLoader m_camera_loader;
};


/// @brief Create dataset object
/// @param dataset_type Type of dataset to create
std::unique_ptr<DatasetBase> create_dataset(const std::string& dataset_type);

} // namespace tinygs