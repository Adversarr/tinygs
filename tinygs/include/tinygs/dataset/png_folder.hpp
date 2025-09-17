#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

/// We expect the folder has the following structure:
/// | folder_path/
/// |   <timestamp1>.EXT
/// |   <timestamp2>.EXT
/// |   ...
/// All the timestamps should match those in the extrinsics file.

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
  std::string m_extension{"png"}; // TODO: support other extensions will break the class name.

  // TODO: this class do not support undistortion.

  /// Loaded data
  ImageShape m_image_shape;
  uint8_t* m_data;
  std::unordered_map<uuid_t, uint8_t*> m_timestamp_data;
  size_t m_size;
};

} // namespace tinygs