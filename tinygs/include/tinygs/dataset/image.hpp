#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/core/pointcloud.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

/// @brief Image folder dataset that natively reads the output of
///   `convert_mipnerf360_to_data_storage.py`.
///
/// Expected directory layout under `root_path`:
///   root_path/
///     cameras.json      -- array of {camera_id, model, width, height, params}
///     poses.json        -- array of {image_id, camera_id, name, qvec, tvec}
///     images/           -- actual image files referenced by `name` in poses.json
///       <name1>
///       <name2>
///       ...
///     points3d.ply      -- (optional) initial point cloud for Gaussian initialization
///
/// All images are loaded into CUDA pinned memory in CHW-tiled uint8 layout.
/// The dataset provides random-access via operator[], returning camera
/// extrinsics, intrinsics, and a non-owning pointer to the image buffer.
class ImageDataset final : public DatasetBase {
public:
  ImageDataset();
  explicit ImageDataset(const std::string& root_path);
  ~ImageDataset() override;

  ImageDataset(const ImageDataset&) = delete;
  ImageDataset& operator=(const ImageDataset&) = delete;

  // -- DatasetBase interface --------------------------------------------------

  void load() override;
  Data operator[](size_t index) const override;
  size_t size() const noexcept override;
  ImageShape image_shape() const override;

  /// @brief Returns the initial point cloud from `points3d.ply` if it exists
  ///        under root_path, or std::nullopt otherwise.
  std::optional<PointCloud> get_point_cloud() const override;

  void set_params(const json& j) override;
  json get_params() const override;

private:
  /// Root directory for this dataset split (e.g. "data/garden/train/").
  std::string m_root_path{"."};

  /// Image file extension filter. Empty means auto-detect from poses.json names.
  std::string m_extension{""};

  /// Resolution mode using reference 3DGS semantics:
  ///   -1: auto cap width to 1600, {1,2,4,8}: divisor, >0: target width.
  int m_resolution{-1};

  /// Additional divisor scale applied to m_resolution logic.
  float m_resolution_scale{1.0f};

  // -- Loaded state -----------------------------------------------------------
  ImageShape m_image_shape{};
  uint8_t* m_data{nullptr};
  std::unordered_map<uuid_t, uint8_t*> m_timestamp_data;
  size_t m_size{0};
};

}  // namespace tinygs
