#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

/// @brief PNG folder dataset with in-memory and pinned memory optimization
class PngFolderDataset final : public DatasetBase {
public:
  PngFolderDataset();

  /**
   * @brief Constructor with automatic image shape inference
   * @param folder_path Path to PNG images folder
   * @param extrinsics_file_path Camera extrinsics file path
   * @param intrinsics_file_path Camera intrinsics file path
   */
  explicit PngFolderDataset(const std::string &folder_path,
                            const std::string &extrinsics_file_path,
                            const std::string &intrinsics_file_path);
  virtual ~PngFolderDataset();

  PngFolderDataset(const PngFolderDataset&) = delete;
  PngFolderDataset& operator=(const PngFolderDataset&) = delete;

  void load() override;
  Data operator[](size_t index) const override;
  size_t size() const noexcept override;
  ImageShape image_shape() const override;

  /// @brief Set parameters from JSON
  void set_params(const json& j) override;
  /// @brief Get parameters as JSON
  json get_params() const override;

private:
  /// Configurables
  std::string m_folder_path{"YOUR_FOLDER_PATH"};
  std::string m_extrinsics_file_path{"YOUR_EXTRINSICS_FILE_PATH"};
  std::string m_intrinsics_file_path{"YOUR_INTRINSICS_FILE_PATH"};

  /// Loaded data
  std::vector<std::string> m_image_paths;
  ImageShape m_image_shape;
  uint8_t* m_data;
  size_t m_size;
};

} // namespace tinygs