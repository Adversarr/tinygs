#include "tinygs/core/camera_loader.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include "tinygs/core/camera_ext.hpp"
#include "tinygs/utils/file.hpp"

namespace tinygs {

SingleCameraLoader::SingleCameraLoader(const std::string& extrinsics_file_path,
                                       const std::string& intrinsics_file_path) {
  load_camera_extrinsics(extrinsics_file_path);
  load_camera_intrinsics(intrinsics_file_path);
}

void SingleCameraLoader::load_camera_extrinsics(const std::string& extrinsics_file_path) {
  auto lines = readlines(extrinsics_file_path);
  m_camera_extrinsics.clear();
  m_camera_extrinsics.reserve(lines.size());

  for (const auto& line : lines) {
    const auto& ext = m_camera_extrinsics.emplace_back(CameraExtrinsics::parse(line));
  }
  std::sort(m_camera_extrinsics.begin(), m_camera_extrinsics.end(), [](const auto& a, const auto& b) {
    return a.frame_idx < b.frame_idx;
  });

  log_info("Loaded {} camera extrinsics from file: {}", m_camera_extrinsics.size(), extrinsics_file_path);
}

void SingleCameraLoader::load_camera_intrinsics(const std::string& intrinsics_file_path) {
  auto lines = readlines(intrinsics_file_path);
  if (lines.empty()) {
    throw std::runtime_error("No camera intrinsics found in file: " + intrinsics_file_path);
  }

  m_camera_intrinsics = CameraIntrinsics::parse(lines[0]);
  log_info("Loaded camera intrinsics from file: {}", intrinsics_file_path);
}

void SingleCameraLoader::resize_sensor(uint32_t width, uint32_t height) {
  if (width == 0 || height == 0) {
    throw std::runtime_error("resize_sensor: width and height must be > 0");
  }
  auto &intr = m_camera_intrinsics;
  if (intr.width == 0 || intr.height == 0) {
    throw std::runtime_error(fmt::format("resize_sensor: intrinsics not initialized (width={}, height={})",
                                         intr.width, intr.height));
  }
  if (intr.width == static_cast<int>(width) && intr.height == static_cast<int>(height)) {
    return; // no change
  }

  float sx = static_cast<float>(width) / static_cast<float>(intr.width);
  float sy = static_cast<float>(height) / static_cast<float>(intr.height);

  float rel_diff = std::fabs(sx - sy) / std::max(sx, sy);
  if (rel_diff > 1e-2f) {
    log_warning("resize_sensor: aspect ratio change detected (scales differ): {:.6f}", rel_diff);
  }

  // Use sx (≈ sy) as scale
  intr.fx *= sx;
  intr.fy *= sy;
  intr.cx *= sx;
  intr.cy *= sy;

  intr.width  = static_cast<int>(width);
  intr.height = static_cast<int>(height);

  log_info("Resized camera sensor to {}x{} (scale {:.6f})", width, height, sx);
}

void SingleCameraLoader::interpolate_to_support(uuid_t frame_idx, uuid_t timestamp) {
  if (std::find_if(m_camera_extrinsics.begin(), m_camera_extrinsics.end(), [timestamp](const auto& ext) {
        return ext.timestamp == timestamp;
      }) != m_camera_extrinsics.end()) {
    // timestamp is already in the support
    return;
  }
  CameraExtrinsics final;
  final.frame_idx = frame_idx;
  final.timestamp = timestamp;
  
  // find the first camera with timestamp >= timestamp
  auto first_larger = std::lower_bound(m_camera_extrinsics.begin(), m_camera_extrinsics.end(), timestamp,
                                       [](const auto& a, uuid_t b) {
                                         return a.timestamp < b;
                                       });
  if (first_larger == m_camera_extrinsics.begin()) {
    // timestamp is smaller than the first camera
    final.m_q = first_larger->m_q;
    final.m_t = first_larger->m_t;
    m_camera_extrinsics.insert(first_larger, final);
  } else if (first_larger == m_camera_extrinsics.end()) {
    // timestamp is larger than the last camera
    final.m_q = m_camera_extrinsics.back().m_q;
    final.m_t = m_camera_extrinsics.back().m_t;
    m_camera_extrinsics.emplace_back(final);
  } else {
    // interpolate between first_larger - 1 and first_larger
    auto lo = first_larger - 1;
    auto hi = first_larger;
    float t = static_cast<float>(timestamp - lo->timestamp) / (hi->timestamp - lo->timestamp);

    auto interp = interpolate(*lo, *hi, t);
    final.m_q = interp.m_q;
    final.m_t = interp.m_t;
    m_camera_extrinsics.insert(first_larger, final);
  }
}

}  // namespace tinygs