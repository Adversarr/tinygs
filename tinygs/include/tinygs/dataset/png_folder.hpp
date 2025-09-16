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
  PngFolderDataset();

  /**
   * @brief Constructor that infers image shape from the first image in the folder
   * @param folder_path Path to the folder containing PNG images
   * @param extrinsics_file_path Path to camera extrinsics file
   * @param intrinsics_file_path Path to camera intrinsics file
   */
  explicit PngFolderDataset(const std::string &folder_path,
                            const std::string &extrinsics_file_path,
                            const std::string &intrinsics_file_path);
  virtual ~PngFolderDataset();

  PngFolderDataset(const PngFolderDataset&) = delete;
  PngFolderDataset& operator=(const PngFolderDataset&) = delete;
  PngFolderDataset(PngFolderDataset&&) noexcept;
  PngFolderDataset& operator=(PngFolderDataset&&) noexcept;

  void load() override;
  Data operator[](size_t index) const override;
  size_t size() const noexcept override;
  ImageShape image_shape() const override;

  SingleCameraLoader &get_camera_loader() noexcept override;

  /// Set dataset parameters from JSON (folder_path, extrinsics_file_path, intrinsics_file_path)
  void set_params(const json& j) override;
  /// Get current dataset parameters as JSON object
  json get_params() const override;

private:
  /// Configurables
  std::string m_folder_path{"YOUR_FOLDER_PATH"};
  std::string m_extrinsics_file_path{"YOUR_EXTRINSICS_FILE_PATH"};
  std::string m_intrinsics_file_path{"YOUR_INTRINSICS_FILE_PATH"};

  /// Loaded data
  std::vector<std::string> m_image_paths;
  SingleCameraLoader m_camera_loader;
  ImageShape m_image_shape;
  uint8_t* m_data;
  size_t m_size;
};

} // namespace tinygs