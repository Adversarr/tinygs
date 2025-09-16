#include "tinygs/dataset/video.hpp"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <stdexcept>

#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "utils/scope_timer.hpp"

namespace tinygs {

VideoDataset::VideoDataset() 
    : m_data(nullptr), m_size(0) {
}

VideoDataset::VideoDataset(const std::string &video_file_path,
                           const std::string &extrinsics_file_path,
                           const std::string &intrinsics_file_path)
    : m_video_file_path(video_file_path), m_extrinsics_file_path(extrinsics_file_path),
      m_intrinsics_file_path(intrinsics_file_path), m_data(nullptr), m_size(0) {
  VideoDataset::load();
}

void VideoDataset::load() {
  TINYGS_TIMER("VideoDataset::load");
  auto start = std::chrono::steady_clock::now();
  
  // Initialize camera loader with stored paths
  m_camera_loader = SingleCameraLoader(m_extrinsics_file_path, m_intrinsics_file_path);
  
  // Open video file
  cv::VideoCapture cap(m_video_file_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("Failed to open video file: " + m_video_file_path);
  }

  // Get video properties
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  m_image_shape = ImageShape(video_width, video_height, 3);

  double fps = cap.get(cv::CAP_PROP_FPS);

  log_info("Video properties: {}x{}, {} frames, {:.2f} fps", video_width, video_height, total_frames, fps);
  
  m_size = std::min(static_cast<size_t>(total_frames), m_camera_loader.get_camera_extrinsics().size());
  if (m_size == 0) {
    throw std::runtime_error("No frames found in video: " + m_video_file_path);
  }

  if (m_size != m_camera_loader.get_camera_extrinsics().size()) {
    log_warning("Number of video frames ({}) and camera extrinsics ({}) do not match.", m_size, m_camera_loader.get_camera_extrinsics().size());
  } else if (m_size != static_cast<size_t>(total_frames)) {
    log_warning("Number of video frames ({}) and camera extrinsics ({}) do not match.", m_size, total_frames);
  }

  // Allocate pinned memory for all frames
  const size_t total_size = m_size * m_image_shape.height * m_image_shape.width * m_image_shape.channel * sizeof(uint8_t);
  CUDA_CHECK_THROW(cudaMallocHost(&m_data, total_size));

  // Load frames sequentially leveraging sorted property of camera extrinsics
  // Since camera extrinsics are sorted by frame_uid, we can load frames sequentially
  // without seeking, which is much faster for video files
  const auto& camera_extrinsics = m_camera_loader.get_camera_extrinsics();

  // Reset video to beginning for sequential reading
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  uint32_t current_video_frame = 0;
  cv::Mat frame;

  for (size_t i = 0; i < m_size; ++i) {
    // frame_uid is 1-based, convert to 0-based for video frame indexing
    uint32_t target_frame_index = camera_extrinsics[i].frame_uid - 1;

    // Validate frame index is within video bounds
    if (target_frame_index >= static_cast<uint32_t>(total_frames)) {
      throw std::runtime_error("Frame UID " + std::to_string(camera_extrinsics[i].frame_uid) + 
                              " exceeds video frame count (" + std::to_string(total_frames) + ")");
    }

    // Read frames sequentially until we reach the target frame
    while (current_video_frame <= target_frame_index) {
      if (!cap.read(frame)) {
        throw std::runtime_error("Failed to read frame " + std::to_string(current_video_frame) + " from video");
      }
      current_video_frame++;
    }

    // Process the frame we just read (which is target_frame_index)
    // Verify frame dimensions match expected dimensions
    if (static_cast<uint32_t>(frame.cols) != m_image_shape.width || static_cast<uint32_t>(frame.rows) != m_image_shape.height) {
      // Resize frame to match expected dimensions
      cv::Mat resized_frame;
      cv::resize(frame, resized_frame, cv::Size(m_image_shape.width, m_image_shape.height));
      frame = resized_frame;
    }

    // Convert from HWC to CHW format and BGR to RGB simultaneously
    uint8_t* dest_ptr = m_data + i * m_image_shape.height * m_image_shape.width * m_image_shape.channel;

    // frame is in HWC format with 3 channels (BGR)
    // dest_ptr should be in CHW format with 3 channels (RGB)
    for (uint32_t c = 0; c < 3; ++c) {
      for (uint32_t h = 0; h < m_image_shape.height; ++h) {
        for (uint32_t w = 0; w < m_image_shape.width; ++w) {
          // Source: HWC format with 3 channels (BGR)
          cv::Vec3b pixel = frame.at<cv::Vec3b>(h, w);
          // Destination: CHW format with 3 channels (RGB) - convert BGR to RGB by reversing channel order
          uint32_t dst_idx = c * m_image_shape.height * m_image_shape.width + h * m_image_shape.width + w;
          dest_ptr[dst_idx] = pixel[2 - c]; // BGR to RGB: B(0)->R(2), G(1)->G(1), R(2)->B(0)
        }
      }
    }
  }
  
  cap.release();
  auto end = std::chrono::steady_clock::now();

  log_info("Loaded {} frames with resolution={}x{}. (consumed {:.6f} GiB in {:.6f} sec.)",
            m_size, m_image_shape.width, m_image_shape.height, static_cast<double>(total_size) / (1024 * 1024 * 1024),
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());

  log_info("Camera Intrinsics: {}", to_string(m_camera_loader.get_camera_intrinsics()));
}

ImageShape VideoDataset::image_shape() const {
  if (!m_data) {
    throw std::runtime_error("Dataset not loaded. Call load() first.");
  }
  return m_image_shape;
}

size_t VideoDataset::size() const noexcept {
  return m_size;
}

SingleCameraLoader &VideoDataset::get_camera_loader() noexcept {
  return m_camera_loader;
}

Data VideoDataset::operator[](size_t index) const {
  if (!m_data) {
    throw std::runtime_error("Dataset not loaded. Call load() first.");
  }
  
  if (index >= m_size) {
    throw std::out_of_range("Index " + std::to_string(index) + " out of range for dataset of size "
                            + std::to_string(m_size));
  }

  Data data;

  // Set up image data
  uint8_t* image_ptr = m_data + index * m_image_shape.height * m_image_shape.width * m_image_shape.channel;
  data.image.shape = image_shape();
  data.image.format = ImageFormat::CHW;  // Converted to CHW format
  data.image.data_type = ImageDataType::UInt8;
  data.image.data = image_ptr;

  // Set camera matrices from camera loader
  data.w2c = m_camera_loader.get_camera_extrinsics()[index].get_w2c();
  data.frame_uid = m_camera_loader.get_camera_extrinsics()[index].frame_uid;
  data.cam_uid = 0; //! assuming single camera
  data.K = m_camera_loader.get_camera_intrinsics().to_mat3();
  return data;
}

VideoDataset::~VideoDataset() {
  if (m_data) {
    CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    m_data = nullptr;
  }
}

VideoDataset::VideoDataset(VideoDataset&& other) noexcept 
    : m_video_file_path(std::move(other.m_video_file_path)),
      m_extrinsics_file_path(std::move(other.m_extrinsics_file_path)),
      m_intrinsics_file_path(std::move(other.m_intrinsics_file_path)),
      m_camera_loader(std::move(other.m_camera_loader)),
      m_image_shape(other.m_image_shape),
      m_data(other.m_data),
      m_size(other.m_size) {
  other.m_data = nullptr;
  other.m_size = 0;
}

VideoDataset& VideoDataset::operator=(VideoDataset&& other) noexcept {
  if (this != &other) {
    // Clean up existing resources
    if (m_data) {
      CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    }

    // Move data from other
    m_video_file_path = std::move(other.m_video_file_path);
    m_extrinsics_file_path = std::move(other.m_extrinsics_file_path);
    m_intrinsics_file_path = std::move(other.m_intrinsics_file_path);
    m_camera_loader = std::move(other.m_camera_loader);
    m_image_shape = other.m_image_shape;
    m_data = other.m_data;
    m_size = other.m_size;
    
    // Reset other
    other.m_data = nullptr;
    other.m_size = 0;
  }
  return *this;
}

void VideoDataset::set_params(const json& j) {
  m_video_file_path = j["video_file_path"].get<std::string>();
  m_extrinsics_file_path = j["extrinsics_file_path"].get<std::string>();
  m_intrinsics_file_path = j["intrinsics_file_path"].get<std::string>();
}

json VideoDataset::get_params() const {
  json params;
  params["type"] = "video";
  params["video_file_path"] = m_video_file_path;
  params["extrinsics_file_path"] = m_extrinsics_file_path;
  params["intrinsics_file_path"] = m_intrinsics_file_path;
  return params;
}

}  // namespace tinygs