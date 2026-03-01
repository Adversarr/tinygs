#pragma once
#include <optional>

#include "tinygs/core/camera_loader.hpp"
#include "tinygs/core/image.hpp"
#include "tinygs/core/pointcloud.hpp"
namespace tinygs {

/// @brief A single training/evaluation sample returned by DatasetBase::operator[].
///
/// `image` is stored in pinned host memory with a CHW-tiled layout (see Image).
/// Camera parameters are in OpenGL convention (right-handed, Y-up).
struct Data {
  Image image;    ///< Image data (UInt8 CHW-tiled, cudaMallocHost pinned memory)

  /// Camera data
  mat4x4 w2c;        ///< World-to-camera 4×4 transform matrix
  mat3x3 K;          ///< Camera intrinsic matrix (3×3)
  uuid_t cam_uid;    ///< Camera ID (0-based, unique per physical camera)
  uuid_t frame_idx;  ///< Frame index within the dataset (1-based)
  uuid_t timestamp;  ///< Unique timestamp / sample ID
};

/// @brief Abstract base class for datasets that supply images + camera poses.
///
/// Lifecycle:
///   1. Construct via `create_dataset(type)`.
///   2. Call `set_params(json)` to configure paths / options.
///   3. Call `load()` to read data from disk into host memory.
///   4. Access samples with `operator[]` and metadata with `size()` / `image_shape()`.
///
/// Thread-safety: `operator[]` is safe to call concurrently from multiple threads
/// after `load()` has completed.
///
/// Implementations: "image" (ImageDataset).
class DatasetBase {
public:
  explicit DatasetBase() = default;
  DatasetBase(const DatasetBase&) = delete;
  DatasetBase& operator=(const DatasetBase&) = delete;
  DatasetBase(DatasetBase&&) = default;
  DatasetBase& operator=(DatasetBase&&) = default;

  /// @brief Load all images and camera data from disk.
  /// @pre `set_params()` must have been called with a valid configuration.
  /// @post `size() > 0` and `operator[]` is usable.
  virtual void load() = 0;

  /// @brief Configure dataset-specific parameters (called before load()).
  virtual void set_params(const json& params) {}

  /// @brief Serialize current parameters to JSON.
  virtual json get_params() const { return json::object(); }

  virtual ~DatasetBase() = default;

  /// @brief Number of samples in the dataset (available after load()).
  virtual size_t size() const noexcept = 0;

  /// @brief Common image shape shared by all samples (width × height × channel).
  ///        The channel count is always 3 (RGB).
  virtual ImageShape image_shape() const = 0;

  /// @brief Access the i-th sample.  Thread-safe after load().
  /// @param idx Sample index in [0, size()).
  virtual Data operator[](size_t idx) const = 0;

  /// @brief Returns the initial point cloud for Gaussian initialization,
  ///        or std::nullopt if the dataset does not provide one (e.g. no points3d.ply).
  virtual std::optional<PointCloud> get_point_cloud() const { return std::nullopt; }

  SingleCameraLoader& get_camera_loader() { return m_camera_loader; }
  const SingleCameraLoader& get_camera_loader() const { return m_camera_loader; }

protected:
  SingleCameraLoader m_camera_loader;
};

/// @brief Factory: create a dataset by type name.
/// @param dataset_type One of: "image".
std::unique_ptr<DatasetBase> create_dataset(const std::string& dataset_type);

} // namespace tinygs