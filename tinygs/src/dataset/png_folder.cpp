#include "tinygs/dataset/png_folder.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>

#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/stbi/stbi_wrapper.h"
#include "utils/scope_timer.hpp"
#include <opencv2/opencv.hpp>

namespace tinygs {

/**
 * @brief Load a single image and convert from RGBA/RGB HWC to CHW+Tiled format in uint8
 * @param index Index of the image in the dataset
 * @param image_path Path to the image file
 * @param data_buffer Pointer to the uint8_t buffer to store the image data
 * @param expected_width Expected width of the image
 * @param expected_height Expected height of the image
 * @param channels Number of channels (should be 3 for RGB)
 */
static void load_single_image(size_t index, const std::string& image_path, uint8_t* data_buffer, 
                              uint32_t expected_width, uint32_t expected_height, uint32_t channels,
                              bool undistort, const cv::Mat& map1, const cv::Mat& map2) {
  auto img = load_stbi_u8(image_path.c_str());

  // TODO: If STB returns images with differing channel counts across files,
  // consider forcing a consistent channel count at load time. Currently we
  // assume RGB output (channels==3) while some PNGs may be RGBA (img.shape.channel==4).

  if (!img.data) {
    throw std::runtime_error("Failed to load image: " + image_path);
  }

  // Verify image dimensions match expected dimensions
  if (static_cast<uint32_t>(img.shape.width) != expected_width ||  //
      static_cast<uint32_t>(img.shape.height) != expected_height) {
    free(img.data);
    throw std::runtime_error("Image dimensions mismatch for " + image_path
                             + ". Expected: " + std::to_string(expected_width) + "x" + std::to_string(expected_height)
                             + ", Got: " + std::to_string(img.shape.width) + "x" + std::to_string(img.shape.height));
  }

  // Use a temporary shape built from expected dimensions and target channels
  // to ensure per-image stride and tiling are consistent with dataset shape.
  ImageShape temp_shape;
  temp_shape.width = expected_width;
  temp_shape.height = expected_height;
  temp_shape.channel = channels; // target output is RGB

  uint8_t* dest_ptr = data_buffer + index * temp_shape.padded_size();
  uint8_t* img_data = (uint8_t*)img.data;

  const auto total_pix = temp_shape.padded_width() * temp_shape.padded_height();

  // Optionally undistort using precomputed maps, then convert to RGB CHW + tiled
  if (undistort) {
    int type = (img.shape.channel == 3) ? CV_8UC3 : CV_8UC4;
    cv::Mat src(expected_height, expected_width, type, img_data);
    cv::Mat undistorted;
    cv::remap(src, undistorted, map1, map2, cv::INTER_LINEAR);

    // Use the undistorted data as source; if 4 channels, ignore alpha
    const int src_channels = undistorted.channels();
    for (uint32_t c = 0; c < channels; ++c) {
      for (uint32_t h = 0; h < expected_height; ++h) {
        for (uint32_t w = 0; w < expected_width; ++w) {
          const auto dst_pix_idx = get_linear_index_tiled(h, w, temp_shape.tiled_width());
          const uint8_t* row_ptr = undistorted.ptr<uint8_t>(h);
          dest_ptr[c * total_pix + dst_pix_idx] = row_ptr[w * src_channels + c];
        }
      }
    }
  } else {
    // img_data is in HWC format with img.shape.channel channels (RGB or RGBA)
    // dest_ptr should be in RGB CHW format (3 channels) + tiled.
    for (uint32_t c = 0; c < channels; ++c) {
      for (uint32_t h = 0; h < expected_height; ++h) {
        for (uint32_t w = 0; w < expected_width; ++w) {
          const auto dst_pix_idx = get_linear_index_tiled(h, w, temp_shape.tiled_width());
          // Source: HWC format; if RGBA, we read only RGB channels (ignore A)
          const uint32_t src_idx = h * expected_width * img.shape.channel + w * img.shape.channel + c;
          dest_ptr[c * total_pix + dst_pix_idx] = img_data[src_idx];
        }
      }
    }
  }

  // Free the temporary image data
  free(img_data);
}

PngFolderDataset::PngFolderDataset() 
    : m_data(nullptr), m_size(0) {
}

PngFolderDataset::PngFolderDataset(const std::string &folder_path,
                                   const std::string &extrinsics_file_path,
                                   const std::string &intrinsics_file_path)
    : m_folder_path(folder_path), m_extrinsics_file_path(extrinsics_file_path),
      m_intrinsics_file_path(intrinsics_file_path), m_data(nullptr), m_size(0) {
  PngFolderDataset::load();
}

static inline std::string get_image(uuid_t timestamp, const std::string& extension, const std::string& folder_path) {
  return fmt::format("{}/{}.{}", folder_path, timestamp, extension);
}

void PngFolderDataset::load() {
  NVTX3_FUNC_RANGE();
  auto start = std::chrono::steady_clock::now();
  
  // Initialize camera loader with stored paths
  m_camera_loader = SingleCameraLoader(m_extrinsics_file_path, m_intrinsics_file_path);

  // Get image paths from folder
  m_size = m_camera_loader.get_camera_extrinsics().size();
  if (m_size == 0) {
    throw std::runtime_error("No PNG files found in folder: " + m_folder_path);
  }

  {
    uuid_t front = m_camera_loader.get_camera_extrinsics().front().timestamp;
    // Infer image shape from the first image
    auto first_img = load_stbi_u8(get_image(front, m_extension, m_folder_path).c_str());
    m_image_shape.width = first_img.shape.width;
    m_image_shape.height = first_img.shape.height;
    m_image_shape.channel = 3;  // Always use 3 channels (RGB) for consistency
    free(first_img.data);       // Free the temporary image data
    // scale the camera to fit the dataset width and height.
    m_camera_loader.resize_sensor(m_image_shape.width, m_image_shape.height);
  }

  // Precompute undistortion maps and update intrinsics if enabled
  cv::Mat map1, map2;
  if (m_undistortion) {
    const auto& intr = m_camera_loader.get_camera_intrinsics();
    cv::Mat K = (cv::Mat_<double>(3, 3) << intr.fx, 0.0, intr.cx,
                                           0.0, intr.fy, intr.cy,
                                           0.0, 0.0, 1.0);
    cv::Mat dist = (cv::Mat_<double>(1, 5) << intr.k1, intr.k2, intr.p1, intr.p2, intr.k3);
    cv::Size image_size(m_image_shape.width, m_image_shape.height);

    cv::Rect valid_roi;
    cv::Mat newK = cv::getOptimalNewCameraMatrix(K, dist, image_size, 0.0, image_size, &valid_roi);
    cv::initUndistortRectifyMap(K, dist, cv::Mat::eye(3, 3, CV_64F), newK, image_size, CV_32FC1, map1, map2);

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

  if (m_image_shape.channel != 3 && m_image_shape.channel != 4) {
    throw std::runtime_error("Only 3 (RGB) or 4 (RGBA) channels are supported now.");
  }

  const size_t per_image = m_image_shape.padded_size();

  // Allocate pinned memory for all images
  const size_t total_size = m_size * per_image * sizeof(uint8_t);
  CUDA_CHECK_THROW(cudaMallocHost(&m_data, total_size));

  // Load all images into memory
#pragma omp parallel for
  for (size_t i = 0; i < m_size; ++i) {
    uuid_t timestamp = m_camera_loader.get_camera_extrinsics().at(i).timestamp;
    std::string image_path = get_image(timestamp, m_extension, m_folder_path);
    load_single_image(i, image_path, m_data, m_image_shape.width, m_image_shape.height, m_image_shape.channel,
                      m_undistortion, map1, map2);
  }
  auto end = std::chrono::steady_clock::now();

  // uuid to data_pointer
  for (size_t i = 0; i < m_size; ++i) {
    uuid_t timestamp = m_camera_loader.get_camera_extrinsics()[i].timestamp;
    m_timestamp_data[timestamp] = m_data + i * per_image;
  }

  log_info("Loaded {} images with resolution={}x{} (inferred from first image). (consumed {:.6f} GiB in {:.6f} sec.)",
            m_size, m_image_shape.width, m_image_shape.height, static_cast<double>(total_size) / (1024 * 1024 * 1024),
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());

  log_info("Camera Intrisics: {}", to_string(m_camera_loader.get_camera_intrinsics()));
}

ImageShape PngFolderDataset::image_shape() const {
  if (!m_data) {
    throw std::runtime_error("Dataset not loaded. Call load() first.");
  }
  return m_image_shape;
}

size_t PngFolderDataset::size() const noexcept {
  return m_size;
}

Data PngFolderDataset::operator[](size_t index) const {
  if (!m_data) {
    throw std::runtime_error("Dataset not loaded. Call load() first.");
  }
  
  if (index >= m_size) {
    throw std::out_of_range(fmt::format("Index {} out of range for dataset of size {}", index, m_size));
  }

  const auto& intrin = m_camera_loader.get_camera_intrinsics();
  const auto& extrin = m_camera_loader.get_camera_extrinsics()[index];

  Data data;
  data.frame_idx = extrin.frame_idx;
  data.cam_uid = 0; // TODO: Support multi-camera video dataset
  data.timestamp = extrin.timestamp;

  // Set up image data
  uint8_t* image_ptr = m_timestamp_data.at(extrin.timestamp);
  data.image.shape = image_shape();
  data.image.data_type = ImageDataType::UInt8;
  data.image.data = image_ptr;

  data.w2c = extrin.get_w2c();
  data.K = intrin.to_mat3();
  return data;
}

PngFolderDataset::~PngFolderDataset() {
  if (m_data) {
    CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    m_data = nullptr;
  }
}

void PngFolderDataset::set_params(const json& j) {
  if (j.contains("folder_path")) {
    m_folder_path = j["folder_path"].get<std::string>();
  }
  if (j.contains("extrinsics_file_path")) {
    m_extrinsics_file_path = j["extrinsics_file_path"].get<std::string>();
  }
  if (j.contains("intrinsics_file_path")) {
    m_intrinsics_file_path = j["intrinsics_file_path"].get<std::string>();
  }
  if (j.contains("extension")) {
    m_extension = j["extension"].get<std::string>();
  }
  if (j.contains("interpolate")) {
    m_interpolate = j["interpolate"].get<bool>();
  }
  if (j.contains("undistortion")) {
    m_undistortion = j["undistortion"].get<bool>();
  }
}

json PngFolderDataset::get_params() const {
  json params;
  params["type"] = "png_folder";
  params["folder_path"] = m_folder_path;
  params["extrinsics_file_path"] = m_extrinsics_file_path;
  params["intrinsics_file_path"] = m_intrinsics_file_path;
  params["extension"] = m_extension;
  params["interpolate"] = m_interpolate;
  params["undistortion"] = m_undistortion;
  return params;
}

}  // namespace tinygs
