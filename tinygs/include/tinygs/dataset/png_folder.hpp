#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

/**
 * @brief Dataset for loading png images from a folder. With in-memory and 
 *        pinned memory optimization.
 */
class PngFolderDataset final : public DatasetBase {
public:
  explicit PngFolderDataset(const std::string &folder_path,
                            const std::string &extrinsics_file_path,
                            const std::string &intrinsics_file_path,
                            const ImageShape &image_shape);
  virtual ~PngFolderDataset();

  PngFolderDataset(const PngFolderDataset&) = delete;
  PngFolderDataset& operator=(const PngFolderDataset&) = delete;
  PngFolderDataset(PngFolderDataset&&) noexcept;
  PngFolderDataset& operator=(PngFolderDataset&&) noexcept;

  Data operator[](size_t index) const override;
  size_t size() const noexcept override;
  ImageShape image_shape() const override;

private:
  std::string m_folder_path;
  std::vector<std::string> m_image_paths;
  SingleCameraLoader m_camera_loader;

  ImageShape m_image_shape;
  uint8_t* m_data;
  size_t m_size;
};

} // namespace tinygs