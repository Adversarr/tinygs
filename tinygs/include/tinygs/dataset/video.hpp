#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

/// @brief Video dataset with in-memory and pinned memory optimization
class VideoDataset final : public DatasetBase {
public:
  VideoDataset();

  /**
   * @brief Constructor with configuration parameters
   * @param video_file_path Video file path
   * @param extrinsics_file_path Camera extrinsics file path
   * @param intrinsics_file_path Camera intrinsics file path
   */
  explicit VideoDataset(const std::string &video_file_path,
                        const std::string &extrinsics_file_path,
                        const std::string &intrinsics_file_path);
  virtual ~VideoDataset();

  VideoDataset(const VideoDataset&) = delete;
  VideoDataset& operator=(const VideoDataset&) = delete;

  void load() override;
  Data operator[](size_t index) const override;
  size_t size() const noexcept override;
  ImageShape image_shape() const override;

  void set_params(const json& j) override;
  json get_params() const override;

private:
  /// Configurables
  std::string m_video_file_path{"YOUR_VIDEO_FILE_PATH"};
  std::string m_extrinsics_file_path{"YOUR_EXTRINSICS_FILE_PATH"};
  std::string m_intrinsics_file_path{"YOUR_INTRINSICS_FILE_PATH"};

  /// Loaded data
  SingleCameraLoader m_camera_loader;
  ImageShape m_image_shape;
  uint8_t* m_data;
  size_t m_size;
};

} // namespace tinygs