#include "tinygs/dataset/png_folder.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>

#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/stbi/stbi_wrapper.h"
#include "utils/scope_timer.hpp"

namespace tinygs {

static std::vector<std::string> list_png_files(const std::string& folder_path) {
  auto result = list_folder(folder_path, true);
  std::vector<std::string> filterd_pngs;
  for (const auto& path : result) {
    if (path.ends_with(".png")) {
      filterd_pngs.push_back(path);
    }
  }

  // sort by name
  std::sort(filterd_pngs.begin(), filterd_pngs.end());

  return filterd_pngs;
}

static void load_single_image(size_t index, const std::string& image_path, float* data_buffer, 
                             uint32_t expected_width, uint32_t expected_height, uint32_t channels) {
  int img_width, img_height;
  float* img_data = load_stbi(&img_width, &img_height, image_path.c_str());

  if (!img_data) {
    throw std::runtime_error("Failed to load image: " + image_path);
  }

  // Verify image dimensions match expected dimensions
  if (static_cast<uint32_t>(img_width) != expected_width || static_cast<uint32_t>(img_height) != expected_height) {
    free(img_data);
    throw std::runtime_error("Image dimensions mismatch for " + image_path
                             + ". Expected: " + std::to_string(expected_width) + "x" + std::to_string(expected_height)
                             + ", Got: " + std::to_string(img_width) + "x" + std::to_string(img_height));
  }

  // Convert from RGBA HWC to RGB CHW format
  float* dest_ptr = data_buffer + index * expected_height * expected_width * channels;
  
  // img_data is in RGBA HWC format (4 channels)
  // dest_ptr should be in RGB CHW format (3 channels)
  for (uint32_t c = 0; c < 3; ++c) {  // Only process RGB channels (skip alpha)
    for (uint32_t h = 0; h < expected_height; ++h) {
      for (uint32_t w = 0; w < expected_width; ++w) {
        // Source: HWC format with 4 channels (RGBA)
        uint32_t src_idx = h * expected_width * 4 + w * 4 + c;
        // Destination: CHW format with 3 channels (RGB)
        uint32_t dst_idx = c * expected_height * expected_width + h * expected_width + w;
        dest_ptr[dst_idx] = img_data[src_idx];
      }
    }
  }

  // Free the temporary image data
  free(img_data);
}



PngFolderDataset::PngFolderDataset(const std::string &folder_path,
                                   const std::string &extrinsics_file_path,
                                   const std::string &intrinsics_file_path,
                                   const ImageShape &image_shape)
    : m_image_paths(list_png_files(folder_path)), m_folder_path(folder_path),
    m_camera_loader(extrinsics_file_path, intrinsics_file_path),
    m_image_shape(image_shape) {
  TINYGS_TIMER("PngFolderDataset::PngFolderDataset");
  auto start = std::chrono::steady_clock::now();
  m_size = std::min(m_image_paths.size(), m_camera_loader.get_camera_extrinsics().size());
  if (m_size == 0) {
    throw std::runtime_error("No PNG files found in folder: " + folder_path);
  }

  if (m_image_shape.channel != 3) {
    throw std::runtime_error("Only 3 channels (RGB) are supported now.");
  }

  if (m_size != m_camera_loader.get_camera_extrinsics().size()) {
    log_warning("Number of PNG files ({}) and camera extrinsics ({}) do not match.", m_size, m_camera_loader.get_camera_extrinsics().size());
  } else if (m_size != m_image_paths.size()) {
    log_warning("Number of PNG files ({}) and camera extrinsics ({}) do not match.", m_size, m_image_paths.size());
  }

  // Allocate pinned memory for all images
  const size_t total_size = m_size * m_image_shape.height * m_image_shape.width * m_image_shape.channel * sizeof(float);
  CUDA_CHECK_THROW(cudaMallocHost(&m_data, total_size));

  // Load all images into memory
#pragma omp parallel for
  for (size_t i = 0; i < m_size; ++i) {
    load_single_image(i, m_image_paths[i], m_data, m_image_shape.width, m_image_shape.height, m_image_shape.channel);
  }
  auto end = std::chrono::steady_clock::now();

  log_info("Loaded {} images with resolution={}x{}. (consumed {:.6f} GiB in {:.6f} sec.)",
            m_size, m_image_shape.width, m_image_shape.height, static_cast<double>(total_size) / (1024 * 1024 * 1024),
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());

  log_info("Camera Intrisics: {}", to_string(m_camera_loader.get_camera_intrinsics()));
}

ImageShape PngFolderDataset::image_shape() const {
  return m_image_shape;
}

size_t PngFolderDataset::size() const noexcept {
  return m_size;
}

Data PngFolderDataset::operator[](size_t index) const {
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

  // Initialize camera matrices to identity (placeholder values)
  // In a real implementation, these would be loaded from camera calibration files
  data.w2c = m_camera_loader.get_camera_extrinsics()[index].get_w2c();
  data.K = m_camera_loader.get_camera_intrinsics().get_K();
  return data;
}

PngFolderDataset::~PngFolderDataset() {
  if (m_data) {
    CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    m_data = nullptr;
  }
}

PngFolderDataset::PngFolderDataset(PngFolderDataset&& other) noexcept {
  m_data = other.m_data;
  other.m_data = nullptr;
}

PngFolderDataset& PngFolderDataset::operator=(PngFolderDataset&& other) noexcept {
  m_data = other.m_data;
  other.m_data = nullptr;
  return *this;
}

}  // namespace tinygs
