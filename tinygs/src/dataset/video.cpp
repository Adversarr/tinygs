#include "tinygs/dataset/video.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "utils/scope_timer.hpp"

namespace tinygs {

static std::unordered_map<uuid_t, uuid_t> load_video_info(const std::string& video_info_path) {
  std::unordered_map<uuid_t, uuid_t> frame_info_list;
  std::ifstream infile(video_info_path);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open video info file: " + video_info_path);
  }

  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    uuid_t frame_uid, timestamp;
    if (!(iss >> frame_uid >> timestamp)) {
      throw std::runtime_error("Malformed line in video info file: " + line);
    }
    // frame_info_list[frame_uid] = timestamp;
    frame_info_list[timestamp] = frame_uid;
  }

  return frame_info_list;
}


VideoDataset::VideoDataset() 
    : m_data(nullptr), m_size(0) {
}

VideoDataset::VideoDataset(const std::string &video_file_path,
                           const std::string &video_info_path,
                           const std::string &extrinsics_file_path,
                           const std::string &intrinsics_file_path)
    : m_video_file_path(video_file_path), m_video_info_path(video_info_path),
      m_extrinsics_file_path(extrinsics_file_path), m_intrinsics_file_path(intrinsics_file_path),
      m_data(nullptr), m_size(0) {
  VideoDataset::load();
}

void VideoDataset::load() {
  TINYGS_TIMER("VideoDataset::load");
  auto start = std::chrono::steady_clock::now();

  // Initialize camera loader with stored paths
  m_camera_loader = SingleCameraLoader(m_extrinsics_file_path, m_intrinsics_file_path);
  m_timestamp_frame = load_video_info(m_video_info_path);
  m_size = m_camera_loader.get_camera_extrinsics().size();
  if (m_size == 0) {
    throw std::runtime_error("No frames found in video: " + m_video_file_path);
  }

  // Open video file
  cv::VideoCapture cap(m_video_file_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("Failed to open video file: " + m_video_file_path);
  }

  // Get video properties
  const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = cap.get(cv::CAP_PROP_FPS);
  m_image_shape = ImageShape(video_width, video_height, 3);

  // NOTE: I do not know why, but the camera loader does not work with the video.
  // I have to resize the sensor to match the video size.
  m_camera_loader.resize_sensor(video_width, video_height);

  if (total_frames != m_timestamp_frame.size()) {
    log_warning("Video frame count {} does not match lines in video info file {}.", total_frames, m_timestamp_frame.size());
  }

  log_info("Video properties: {}x{}, {} frames, {:.2f} fps", video_width, video_height, total_frames, fps);

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
    // NOTE: The extrinsics ensures the list is sorted.
    const uuid_t target_timestamp = camera_extrinsics[i].timestamp;
    const uuid_t target_video_frame_index = m_timestamp_frame.at(target_timestamp) - 1;

    log_debug("Load {}/{}: frame_idx={}, timestamp={}, video_frame_idx={}", i + 1, m_size,  //
              camera_extrinsics[i].frame_idx, target_timestamp, target_video_frame_index);

    // Validate frame index is within video bounds
    if (target_video_frame_index >= static_cast<uint32_t>(total_frames)) {
      throw std::runtime_error("Frame UID " + std::to_string(target_video_frame_index)
                               + " exceeds video frame count (" + std::to_string(total_frames) + ")");
    }

    // Read frames sequentially until we reach the target frame
    while (current_video_frame <= target_video_frame_index) {
      if (!cap.read(frame)) {
        throw std::runtime_error("Failed to read frame " + std::to_string(current_video_frame) + " from video");
      }
      current_video_frame++;
    }

    // Process the frame we just read (which is target_frame_index)
    // Verify frame dimensions match expected dimensions
    if (static_cast<uint32_t>(frame.cols) != m_image_shape.width
        || static_cast<uint32_t>(frame.rows) != m_image_shape.height) {
      // Resize frame to match expected dimensions
      cv::Mat resized_frame;
      cv::resize(frame, resized_frame, cv::Size(m_image_shape.width, m_image_shape.height));
      frame = resized_frame;
    }

    // Process frame in HWC format (RGB) - no format conversion needed
    uint8_t* dest_ptr = m_data + i * m_image_shape.height * m_image_shape.width * m_image_shape.channel;

    // frame is in HWC format with 3 channels (BGR)
    // dest_ptr will be in HWC format with 3 channels (RGB) - convert BGR to RGB
    for (uint32_t h = 0; h < m_image_shape.height; ++h) {
      for (uint32_t w = 0; w < m_image_shape.width; ++w) {
        for (uint32_t c = 0; c < 3; ++c) {
          // Source: HWC format with 3 channels (BGR)
          cv::Vec3b pixel = frame.at<cv::Vec3b>(h, w);
          // Destination: HWC format with 3 channels (RGB) - convert BGR to RGB by reversing channel order
          uint32_t dst_idx = h * m_image_shape.width * m_image_shape.channel + w * m_image_shape.channel + c;
          dest_ptr[dst_idx] = pixel[2 - c]; // BGR to RGB: B(0)->R(2), G(1)->G(1), R(2)->B(0)
        }
      }
    }

    m_timestamp_data[target_timestamp] = dest_ptr;
  }

  // Release video capture resources
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

Data VideoDataset::operator[](size_t index) const {
  if (!m_data) {
    throw std::runtime_error("Dataset not loaded. Call load() first.");
  }
  
  if (index >= m_size) {
    throw std::out_of_range("Index " + std::to_string(index) + " out of range for dataset of size "
                            + std::to_string(m_size));
  }

  const auto& intrin = m_camera_loader.get_camera_intrinsics();
  const auto& extrin = m_camera_loader.get_camera_extrinsics()[index];
  const auto timestamp = extrin.timestamp;

  Data data;
  data.frame_idx = extrin.frame_idx;
  data.cam_uid = 0; // TODO: Support multi-camera video dataset
  data.timestamp = timestamp;

  // Set up image data
  uint8_t* image_ptr = m_timestamp_data.at(timestamp);
  data.image.shape = image_shape();
  data.image.data_type = ImageDataType::UInt8;
  data.image.data = image_ptr;

  // Set camera matrices from camera loader
  data.w2c = extrin.get_w2c();
  data.K = intrin.to_mat3();
  return data;
}

VideoDataset::~VideoDataset() {
  if (m_data) {
    CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    m_data = nullptr;
  }
}

void VideoDataset::set_params(const json& j) {
  m_video_file_path = j["video_file_path"].get<std::string>();
  m_video_info_path = j["video_info_path"].get<std::string>();
  m_extrinsics_file_path = j["extrinsics_file_path"].get<std::string>();
  m_intrinsics_file_path = j["intrinsics_file_path"].get<std::string>();
}

json VideoDataset::get_params() const {
  json params;
  params["type"] = "video";
  params["video_file_path"] = m_video_file_path;
  params["video_info_path"] = m_video_info_path;
  params["extrinsics_file_path"] = m_extrinsics_file_path;
  params["intrinsics_file_path"] = m_intrinsics_file_path;
  return params;
}

}  // namespace tinygs