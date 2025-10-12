#include "tinygs/dataset/video.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "utils/image_format.hpp"
#include "utils/scope_timer.hpp"

namespace tinygs {

static std::unordered_map<uuid_t, uuid_t> load_video_info(const std::string& video_info_path) {
  std::unordered_map<uuid_t, uuid_t> frame_info_list;
  std::ifstream infile(video_info_path);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open video info file: " + video_info_path);
  }

  std::string line;
  uuid_t last_uid = 0;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    uuid_t frame_uid, timestamp, unuse;
    if (!(iss >> frame_uid >> timestamp >> unuse)) {
      throw std::runtime_error("Malformed line in video info file: " + line);
    }
    if (frame_info_list.find(timestamp) != frame_info_list.end()) {
      log_error("Duplicate timestamp in video info file: " + std::to_string(timestamp));
    }

    frame_info_list[timestamp] = frame_uid;
    last_uid = frame_uid;
  }

  if (last_uid != frame_info_list.size() - 1) {
    log_error("Frame uid {} does not match expected index {}", last_uid, frame_info_list.size() - 1);
  }

  return frame_info_list;
}

/**
 * @brief Load a single video frame and convert from BGR HWC to RGB CHW+Tiled format in uint8
 * @param index Index of the frame in the dataset
 * @param frame OpenCV Mat containing the video frame in BGR HWC format
 * @param data_buffer Pointer to the uint8_t buffer to store the frame data
 * @param expected_width Expected width of the frame
 * @param expected_height Expected height of the frame
 * @param channels Number of channels (should be 3 for RGB)
 */
static void load_single_frame(size_t index, const cv::Mat& frame, uint8_t* data_buffer, 
                              uint32_t expected_width, uint32_t expected_height, uint32_t channels) {
  if (static_cast<uint32_t>(frame.cols) != expected_width || 
      static_cast<uint32_t>(frame.rows) != expected_height) {
    throw std::runtime_error("Frame dimensions mismatch. Expected: " + 
                             std::to_string(expected_width) + "x" + std::to_string(expected_height) +
                             ", Got: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows));
  }

  // Calculate padded image shape for tile-based storage
  ImageShape temp_shape;
  temp_shape.width = expected_width;
  temp_shape.height = expected_height;
  temp_shape.channel = channels;
  
  uint8_t* dest_ptr = data_buffer + index * temp_shape.padded_size();
  auto total_pix = temp_shape.padded_width() * temp_shape.padded_height();

  // frame is in BGR HWC format (3 channels)
  // dest_ptr should be in RGB CHW format (3 channels) + tiled.
  for (uint32_t c = 0; c < channels; ++c) {
    for (uint32_t h = 0; h < expected_height; ++h) {
      for (uint32_t w = 0; w < expected_width; ++w) {
        const auto dst_pix_idx = get_linear_index_tiled(h, w, temp_shape.tiled_width());
        // Source: HWC format with 3 channels (BGR)
        cv::Vec3b pixel = frame.at<cv::Vec3b>(h, w);
        // Destination: CHW format with 3 channels (RGB) - convert BGR to RGB
        // BGR to RGB: B(0)->R(2), G(1)->G(1), R(2)->B(0)
        dest_ptr[c * total_pix + dst_pix_idx] = pixel[2 - c];
      }
    }
  }
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
  NVTX3_FUNC_RANGE();
  auto start = std::chrono::steady_clock::now();

  // Initialize camera loader with stored paths
  m_camera_loader = SingleCameraLoader(m_extrinsics_file_path, m_intrinsics_file_path);
  m_timestamp_frame = load_video_info(m_video_info_path);
  
  if (m_interpolate){
    for (auto & [timestamp, frame_idx] : m_timestamp_frame) {
      m_camera_loader.interpolate_to_support(frame_idx, timestamp);
    }
  }
  m_size = m_camera_loader.get_camera_extrinsics().size();
  if (m_size == 0) {
    throw std::runtime_error("No frames found in video: " + m_video_file_path);
  }
  log_info("Loaded {} frames from video: {}", m_size, m_video_file_path);

  // Open video file
  cv::VideoCapture cap(m_video_file_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("Failed to open video file: " + m_video_file_path);
  }

  cv::Mat map1, map2;
  if (m_undistortion) {
    // Query video resolution and align intrinsics to it
    const int video_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int video_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_camera_loader.resize_sensor(video_width, video_height);

    // Build OpenCV camera matrix and distortion coefficients
    const auto& intr = m_camera_loader.get_camera_intrinsics();
    cv::Mat K = (cv::Mat_<double>(3, 3) << intr.fx, 0.0, intr.cx,
                                           0.0, intr.fy, intr.cy,
                                           0.0, 0.0, 1.0);
    cv::Mat dist = (cv::Mat_<double>(1, 5) << intr.k1, intr.k2, intr.p1, intr.p2, intr.k3);
    cv::Size image_size(intr.width, intr.height);

    // Compute optimal new camera matrix (alpha=0 to minimize black regions)
    cv::Rect valid_roi;
    cv::Mat newK = cv::getOptimalNewCameraMatrix(K, dist, image_size, 0.0, image_size, &valid_roi);

    // Initialize undistortion map
    cv::initUndistortRectifyMap(K, dist, cv::Mat::eye(3, 3, CV_64F), newK, image_size, CV_32FC1, map1, map2);

    // Update intrinsics to the new camera matrix and zero distortion (since frames will be undistorted)
    CameraIntrinsics new_intrisics = intr;
    new_intrisics.fx = static_cast<float>(newK.at<double>(0, 0));
    new_intrisics.fy = static_cast<float>(newK.at<double>(1, 1));
    new_intrisics.cx = static_cast<float>(newK.at<double>(0, 2));
    new_intrisics.cy = static_cast<float>(newK.at<double>(1, 2));
    new_intrisics.k1 = 0.0f;
    new_intrisics.k2 = 0.0f;
    new_intrisics.k3 = 0.0f;
    new_intrisics.p1 = 0.0f;
    new_intrisics.p2 = 0.0f;
    m_camera_loader.set_camera_intrinsics(new_intrisics);
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

  // Allocate pinned memory for all frames using padded_size for tile-based storage
  const size_t total_size = m_size * m_image_shape.padded_size();
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

    // Optionally undistort then convert from BGR HWC to RGB CHW+Tiled format
    if (m_undistortion) {
      cv::Mat undistorted;
      cv::remap(frame, undistorted, map1, map2, cv::INTER_LINEAR);
      load_single_frame(i, undistorted, m_data, m_image_shape.width, m_image_shape.height, m_image_shape.channel);
    } else {
      load_single_frame(i, frame, m_data, m_image_shape.width, m_image_shape.height, m_image_shape.channel);
    }

    m_timestamp_data[target_timestamp] = m_data + i * m_image_shape.padded_size();
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
  if (j.contains("interpolate")) {
    m_interpolate = j["interpolate"].get<bool>();
  }
  if (j.contains("video_file_path")) {
    m_video_file_path = j["video_file_path"].get<std::string>();
  }
  if (j.contains("video_info_path")) {
    m_video_info_path = j["video_info_path"].get<std::string>();
  }
  if (j.contains("extrinsics_file_path")) {
    m_extrinsics_file_path = j["extrinsics_file_path"].get<std::string>();
  }
  if (j.contains("intrinsics_file_path")) {
    m_intrinsics_file_path = j["intrinsics_file_path"].get<std::string>();
  }
  if (j.contains("undistortion")) {
    m_undistortion = j["undistortion"].get<bool>();
  }
}

json VideoDataset::get_params() const {
  json params;
  params["type"] = "video";
  params["video_file_path"] = m_video_file_path;
  params["video_info_path"] = m_video_info_path;
  params["extrinsics_file_path"] = m_extrinsics_file_path;
  params["intrinsics_file_path"] = m_intrinsics_file_path;
  params["interpolate"] = m_interpolate;
  params["undistortion"] = m_undistortion;
  return params;
}

}  // namespace tinygs