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

static void load_single_frame(size_t index, cv::VideoCapture& cap, float* data_buffer, 
                             uint32_t expected_width, uint32_t expected_height, uint32_t channels) {
  // Seek to the specific frame
  cap.set(cv::CAP_PROP_POS_FRAMES, index);
  
  cv::Mat frame;
  if (!cap.read(frame)) {
    throw std::runtime_error("Failed to read frame " + std::to_string(index) + " from video");
  }

  // Verify frame dimensions match expected dimensions
  if (static_cast<uint32_t>(frame.cols) != expected_width || static_cast<uint32_t>(frame.rows) != expected_height) {
    // Resize frame to match expected dimensions
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(expected_width, expected_height));
    frame = resized_frame;
  }

  // Convert BGR to RGB and normalize to float [0,1]
  cv::Mat rgb_frame;
  cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
  
  // Convert to float and normalize
  cv::Mat float_frame;
  rgb_frame.convertTo(float_frame, CV_32F, 1.0/255.0);

  // Convert from HWC to CHW format
  float* dest_ptr = data_buffer + index * expected_height * expected_width * channels;
  
  // float_frame is in HWC format with 3 channels (RGB)
  // dest_ptr should be in CHW format with 3 channels (RGB)
  for (uint32_t c = 0; c < 3; ++c) {
    for (uint32_t h = 0; h < expected_height; ++h) {
      for (uint32_t w = 0; w < expected_width; ++w) {
        // Source: HWC format with 3 channels (RGB)
        cv::Vec3f pixel = float_frame.at<cv::Vec3f>(h, w);
        // Destination: CHW format with 3 channels (RGB)
        uint32_t dst_idx = c * expected_height * expected_width + h * expected_width + w;
        dest_ptr[dst_idx] = pixel[c];
      }
    }
  }
}

VideoDataset::VideoDataset(const std::string &video_file_path,
                           const std::string &extrinsics_file_path,
                           const std::string &intrinsics_file_path,
                           const ImageShape &image_shape)
    : m_video_file_path(video_file_path),
      m_camera_loader(extrinsics_file_path, intrinsics_file_path),
      m_image_shape(image_shape) {
  TINYGS_TIMER("VideoDataset::VideoDataset");
  auto start = std::chrono::steady_clock::now();
  
  // Open video file
  cv::VideoCapture cap(video_file_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("Failed to open video file: " + video_file_path);
  }

  // Get video properties
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  
  log_info("Video properties: {}x{}, {} frames, {:.2f} fps", video_width, video_height, total_frames, fps);
  
  m_size = std::min(static_cast<size_t>(total_frames), m_camera_loader.get_camera_extrinsics().size());
  if (m_size == 0) {
    throw std::runtime_error("No frames found in video: " + video_file_path);
  }

  if (m_image_shape.channel != 3) {
    throw std::runtime_error("Only 3 channels (RGB) are supported now.");
  }

  if (m_size != m_camera_loader.get_camera_extrinsics().size()) {
    log_warning("Number of video frames ({}) and camera extrinsics ({}) do not match.", m_size, m_camera_loader.get_camera_extrinsics().size());
  } else if (m_size != static_cast<size_t>(total_frames)) {
    log_warning("Number of video frames ({}) and camera extrinsics ({}) do not match.", m_size, total_frames);
  }

  // Allocate pinned memory for all frames
  const size_t total_size = m_size * m_image_shape.height * m_image_shape.width * m_image_shape.channel * sizeof(float);
  CUDA_CHECK_THROW(cudaMallocHost(&m_data, total_size));

  // Load all frames into memory
  for (size_t i = 0; i < m_size; ++i) {
    load_single_frame(i, cap, m_data, m_image_shape.width, m_image_shape.height, m_image_shape.channel);
  }
  
  cap.release();
  auto end = std::chrono::steady_clock::now();

  log_info("Loaded {} frames with resolution={}x{}. (consumed {:.6f} GiB in {:.6f} sec.)",
            m_size, m_image_shape.width, m_image_shape.height, static_cast<double>(total_size) / (1024 * 1024 * 1024),
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());

  log_info("Camera Intrinsics: {}", to_string(m_camera_loader.get_camera_intrinsics()));
}

ImageShape VideoDataset::image_shape() const {
  return m_image_shape;
}

size_t VideoDataset::size() const noexcept {
  return m_size;
}

Data VideoDataset::operator[](size_t index) const {
  if (index >= m_size) {
    throw std::out_of_range("Index " + std::to_string(index) + " out of range for dataset of size "
                            + std::to_string(m_size));
  }

  Data data;

  // Set up image data
  float* image_ptr = m_data + index * m_image_shape.height * m_image_shape.width * m_image_shape.channel;
  data.image.shape = image_shape();
  data.image.format = ImageFormat::CHW;  // Converted to CHW format
  data.image.data = image_ptr;

  // Set camera matrices from camera loader
  data.w2c = m_camera_loader.get_camera_extrinsics()[index].get_w2c();
  data.K = m_camera_loader.get_camera_intrinsics().get_K();
  return data;
}

VideoDataset::~VideoDataset() {
  if (m_data) {
    CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    m_data = nullptr;
  }
}

VideoDataset::VideoDataset(VideoDataset&& other) noexcept {
  m_video_file_path = std::move(other.m_video_file_path);
  m_camera_loader = std::move(other.m_camera_loader);
  m_image_shape = other.m_image_shape;
  m_data = other.m_data;
  m_size = other.m_size;
  other.m_data = nullptr;
}

VideoDataset& VideoDataset::operator=(VideoDataset&& other) noexcept {
  if (this != &other) {
    // Clean up existing resources
    if (m_data) {
      CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    }
    
    // Move from other
    m_video_file_path = std::move(other.m_video_file_path);
    m_camera_loader = std::move(other.m_camera_loader);
    m_image_shape = other.m_image_shape;
    m_data = other.m_data;
    m_size = other.m_size;
    other.m_data = nullptr;
  }
  return *this;
}

}  // namespace tinygs