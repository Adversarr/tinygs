#include "tinygs/dataset/image.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>

#include "tinygs/core/camera.hpp"
#include "tinygs/core/camera_loader.hpp"
#include "tinygs/core/pointcloud.hpp"
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/utils/file.hpp"
#include "tinygs/utils/stbi/stbi_wrapper.h"

namespace tinygs {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a single image from HWC (RGB/RGBA) to CHW-tiled uint8 layout.
static void load_single_image(size_t index, const std::string& image_path, uint8_t* data_buffer,
                              uint32_t expected_width, uint32_t expected_height, uint32_t channels) {
  auto img = load_stbi_u8(image_path.c_str());
  if (!img.data) {
    throw std::runtime_error("Failed to load image: " + image_path);
  }

  if (static_cast<uint32_t>(img.shape.width) != expected_width ||
      static_cast<uint32_t>(img.shape.height) != expected_height) {
    free(img.data);
    throw std::runtime_error("Image dimensions mismatch for " + image_path
                             + ". Expected: " + std::to_string(expected_width) + "x"
                             + std::to_string(expected_height)
                             + ", Got: " + std::to_string(img.shape.width) + "x"
                             + std::to_string(img.shape.height));
  }

  ImageShape temp_shape{expected_width, expected_height, channels};
  uint8_t* dest_ptr = data_buffer + index * temp_shape.padded_size();
  uint8_t* img_data = static_cast<uint8_t*>(img.data);
  const auto total_pix = temp_shape.padded_width() * temp_shape.padded_height();

  // HWC → CHW tiled; ignore alpha channel if present.
  for (uint32_t c = 0; c < channels; ++c) {
    for (uint32_t h = 0; h < expected_height; ++h) {
      for (uint32_t w = 0; w < expected_width; ++w) {
        const auto dst_pix_idx = get_linear_index_tiled(h, w, temp_shape.tiled_width());
        const uint32_t src_idx = h * expected_width * img.shape.channel + w * img.shape.channel + c;
        dest_ptr[c * total_pix + dst_pix_idx] = img_data[src_idx];
      }
    }
  }
  free(img_data);
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

ImageDataset::ImageDataset() = default;

ImageDataset::ImageDataset(const std::string& root_path) : m_root_path(root_path) {
  ImageDataset::load();
}

ImageDataset::~ImageDataset() {
  if (m_data) {
    CUDA_CHECK_PRINT(cudaFreeHost(m_data));
    m_data = nullptr;
  }
}

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

void ImageDataset::load() {
  NVTX3_FUNC_RANGE();
  namespace fs = std::filesystem;
  auto start = std::chrono::steady_clock::now();

  const fs::path root{m_root_path};
  const fs::path cameras_json_path = root / "cameras.json";
  const fs::path poses_json_path = root / "poses.json";
  const fs::path images_dir = root / "images";

  // --- Load camera parameters from JSON --------------------------------------
  if (fs::exists(cameras_json_path) && fs::exists(poses_json_path)) {
    m_camera_loader.load_from_json(cameras_json_path.string(), poses_json_path.string());
  } else {
    throw std::runtime_error(
        "ImageDataset: cameras.json and poses.json are required under root_path: " + m_root_path);
  }

  // --- Build image name lookup from poses.json -------------------------------
  //     The camera loader has the extrinsics sorted by frame_idx.
  //     We need the filename (`name` field) for each pose to locate images.
  nlohmann::json poses_json;
  {
    std::ifstream f(poses_json_path.string());
    poses_json = nlohmann::json::parse(f);
  }

  // Build a map: image_id → name (filename)
  std::unordered_map<uuid_t, std::string> image_id_to_name;
  for (const auto& pose : poses_json) {
    uuid_t id = pose.at("image_id").get<uint64_t>();
    image_id_to_name[id] = pose.at("name").get<std::string>();
  }

  m_size = m_camera_loader.get_camera_extrinsics().size();
  if (m_size == 0) {
    throw std::runtime_error("ImageDataset: no camera extrinsics found in " + poses_json_path.string());
  }

  // --- Infer image shape from the first image --------------------------------
  {
    const auto& first_ext = m_camera_loader.get_camera_extrinsics().front();
    const std::string& first_name = image_id_to_name.at(first_ext.frame_idx);
    const std::string first_path = (images_dir / first_name).string();
    auto first_img = load_stbi_u8(first_path.c_str());
    if (!first_img.data) {
      throw std::runtime_error("ImageDataset: failed to load first image: " + first_path);
    }
    m_image_shape.width = first_img.shape.width;
    m_image_shape.height = first_img.shape.height;
    m_image_shape.channel = 3;  // always RGB
    free(first_img.data);
    m_camera_loader.resize_sensor(m_image_shape.width, m_image_shape.height);
  }

  // --- Allocate pinned memory for all images ---------------------------------
  const size_t per_image = m_image_shape.padded_size();
  const size_t total_size = m_size * per_image * sizeof(uint8_t);
  CUDA_CHECK_THROW(cudaMallocHost(&m_data, total_size));

  // Build ordered list of image paths matching extrinsics order
  std::vector<std::string> image_paths(m_size);
  for (size_t i = 0; i < m_size; ++i) {
    const auto& ext = m_camera_loader.get_camera_extrinsics()[i];
    const std::string& name = image_id_to_name.at(ext.frame_idx);
    image_paths[i] = (images_dir / name).string();
  }

  // --- Load all images in parallel -------------------------------------------
#pragma omp parallel for
  for (size_t i = 0; i < m_size; ++i) {
    load_single_image(i, image_paths[i], m_data, m_image_shape.width, m_image_shape.height,
                      m_image_shape.channel);
  }

  // Build timestamp → data pointer map
  for (size_t i = 0; i < m_size; ++i) {
    uuid_t timestamp = m_camera_loader.get_camera_extrinsics()[i].timestamp;
    m_timestamp_data[timestamp] = m_data + i * per_image;
  }

  auto end = std::chrono::steady_clock::now();
  log_info(
      "ImageDataset: loaded {} images ({}x{}) from '{}'. "
      "Memory: {:.2f} GiB, Time: {:.3f} s.",
      m_size, m_image_shape.width, m_image_shape.height, m_root_path,
      static_cast<double>(total_size) / (1024.0 * 1024.0 * 1024.0),
      std::chrono::duration<double>(end - start).count());
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

ImageShape ImageDataset::image_shape() const {
  if (!m_data) {
    throw std::runtime_error("ImageDataset: not loaded. Call load() first.");
  }
  return m_image_shape;
}

size_t ImageDataset::size() const noexcept { return m_size; }

Data ImageDataset::operator[](size_t index) const {
  if (!m_data) {
    throw std::runtime_error("ImageDataset: not loaded. Call load() first.");
  }
  if (index >= m_size) {
    throw std::out_of_range(fmt::format("ImageDataset: index {} out of range (size {})", index, m_size));
  }

  const auto& extrin = m_camera_loader.get_camera_extrinsics()[index];
  const uuid_t cam_uid = extrin.cam_uid;
  const auto& intrin = m_camera_loader.get_camera_intrinsics()[cam_uid];

  Data data;
  data.frame_idx = extrin.frame_idx;
  data.cam_uid = cam_uid;
  data.timestamp = extrin.timestamp;

  uint8_t* image_ptr = m_timestamp_data.at(extrin.timestamp);
  data.image.shape = m_image_shape;
  data.image.data_type = DataType::UInt8;
  data.image.data = image_ptr;

  data.w2c = extrin.get_w2c();
  data.K = intrin.to_mat3();
  return data;
}

// ---------------------------------------------------------------------------
// Point cloud
// ---------------------------------------------------------------------------

std::optional<PointCloud> ImageDataset::get_point_cloud() const {
  namespace fs = std::filesystem;
  fs::path pc_path = fs::path(m_root_path) / "points3d.ply";
  if (!fs::exists(pc_path)) {
    return std::nullopt;
  }
  return load_point_cloud(pc_path.string());
}

// ---------------------------------------------------------------------------
// JSON params
// ---------------------------------------------------------------------------

void ImageDataset::set_params(const json& j) {
  if (j.contains("root_path")) {
    m_root_path = j["root_path"].get<std::string>();
  }
  if (j.contains("extension")) {
    m_extension = j["extension"].get<std::string>();
  }
}

json ImageDataset::get_params() const {
  json params;
  params["type"] = "image";
  params["root_path"] = m_root_path;
  if (!m_extension.empty()) {
    params["extension"] = m_extension;
  }
  return params;
}

}  // namespace tinygs
