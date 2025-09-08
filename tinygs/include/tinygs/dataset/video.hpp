#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/dataset/dataset.hpp"

namespace tinygs {

/**
 * @brief Dataset for loading frames from MP4 video files. With in-memory and 
 *        pinned memory optimization.
 */
class VideoDataset final : public DatasetBase {
public:
  explicit VideoDataset(const std::string &video_file_path,
                        const std::string &extrinsics_file_path,
                        const std::string &intrinsics_file_path,
                        const ImageShape &image_shape);
  virtual ~VideoDataset();

  VideoDataset(const VideoDataset&) = delete;
  VideoDataset& operator=(const VideoDataset&) = delete;
  VideoDataset(VideoDataset&&) noexcept;
  VideoDataset& operator=(VideoDataset&&) noexcept;

  Data operator[](size_t index) const override;
  size_t size() const noexcept override;
  ImageShape image_shape() const override;

private:
  std::string m_video_file_path;
  SingleCameraIntrinsics m_camera_loader;

  ImageShape m_image_shape;
  float* m_data;
  size_t m_size;
};

} // namespace tinygs