#include "tinygs/dataset/png_folder.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>

#include "tinygs/cuda/common_host.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/stbi/stbi_wrapper.h"

namespace tinygs {

static std::vector<std::string> list_png_files(const std::string& folder_path) {
  auto result = list_folder(folder_path, true);
  std::vector<std::string> filterd_pngs;
  for (const auto& path : result) {
    if (path.ends_with(".png")) {
      filterd_pngs.push_back(path);
    }
  }
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

  // Copy image data to pinned memory
  float* dest_ptr = data_buffer + index * expected_height * expected_width * channels;
  std::memcpy(dest_ptr, img_data, expected_height * expected_width * channels * sizeof(float));

  // Free the temporary image data
  free(img_data);
}

PngFolderDataset::PngFolderDataset(const std::string& folder_path, uint32_t height, uint32_t width) :
    m_image_paths(list_png_files(folder_path)),
    m_folder_path(folder_path),
    m_height(height),
    m_width(width),
    m_channels(4) {
  m_size = m_image_paths.size();
  log_debug("Loading {} images from folder: {}", m_size, folder_path);

  if (m_size == 0) {
    throw std::runtime_error("No PNG files found in folder: " + folder_path);
  }

  // Allocate pinned memory for all images
  size_t total_size = m_size * m_height * m_width * m_channels * sizeof(float);
  CUDA_CHECK_THROW(cudaMallocHost(&m_data, total_size));

  auto start = std::chrono::steady_clock::now();
  // Load all images into memory
  #pragma omp parallel for
  for (size_t i = 0; i < m_size; ++i) {
    load_single_image(i, m_image_paths[i], m_data, m_width, m_height, m_channels);
  }
  auto end = std::chrono::steady_clock::now();

  log_info("Loaded {} images with resolution={}x{}. (consumed {:.6f} GiB in {:.6f} sec.)",
            m_size, width, height, static_cast<double>(total_size) / (1024 * 1024 * 1024),
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
}

Data PngFolderDataset::operator[](size_t index) const {
  if (index >= m_size) {
    throw std::out_of_range("Index " + std::to_string(index) + " out of range for dataset of size "
                            + std::to_string(m_size));
  }

  Data data;

  // Set up image data
  float* image_ptr = m_data + index * m_height * m_width * m_channels;
  data.image.width = static_cast<uint16_t>(m_width);
  data.image.height = static_cast<uint16_t>(m_height);
  data.image.channels = static_cast<uint16_t>(m_channels);
  data.image.format = ImageFormat::HWC;  // stb_image standard
  data.image.data = PitchedPtr<const float>(image_ptr, m_width * m_channels * sizeof(float));

  // Initialize camera matrices to identity (placeholder values)
  // In a real implementation, these would be loaded from camera calibration files
  data.w2c = mat4x4(1.0f);  // Identity matrix
  data.K = mat3x3(1.0f);    // Identity matrix

  return data;
}

PngFolderDataset::~PngFolderDataset() {
  if (m_data) {
    CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    m_data = nullptr;
  }
}

}  // namespace tinygs
