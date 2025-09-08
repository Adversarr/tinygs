#include "tinygs/core/camera_loader.hpp"

#include <stdexcept>

#include "tinygs/utils/file.hpp"

namespace tinygs {

SingleCameraIntrinsics::SingleCameraIntrinsics(const std::string& extrinsics_file_path,
                                               const std::string& intrinsics_file_path) {
  load_camera_extrinsics(extrinsics_file_path);
  load_camera_intrinsics(intrinsics_file_path);
}

void SingleCameraIntrinsics::load_camera_extrinsics(const std::string& extrinsics_file_path) {
  auto lines = readlines(extrinsics_file_path);
  m_camera_extrinsics.clear();
  m_camera_extrinsics.reserve(lines.size());

  for (auto& line : lines) {
    m_camera_extrinsics.emplace_back(CameraExtrinsics::parse(line));
  }

  log_info("Loaded {} camera extrinsics from file: {}", m_camera_extrinsics.size(), extrinsics_file_path);
}

void SingleCameraIntrinsics::load_camera_intrinsics(const std::string& intrinsics_file_path) {
  auto lines = readlines(intrinsics_file_path);
  if (lines.empty()) {
    throw std::runtime_error("No camera intrinsics found in file: " + intrinsics_file_path);
  }

  m_camera_intrinsics = CameraIntrinsics::parse(lines[0]);
  log_info("Loaded camera intrinsics from file: {}", intrinsics_file_path);
}

}  // namespace tinygs