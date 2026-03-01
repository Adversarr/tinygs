#include "tinygs/core/camera_loader.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

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

  m_camera_intrinsics.clear();
  m_camera_intrinsics.reserve(lines.size());
  for (const auto& line : lines) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    m_camera_intrinsics.emplace_back(CameraIntrinsics::parse(line));
  }
  log_info("Loaded {} camera intrinsics from file: {}", m_camera_intrinsics.size(), intrinsics_file_path);
}

void SingleCameraLoader::resize_sensor(uint32_t width, uint32_t height) {
  for (auto& intr : m_camera_intrinsics) {
    if (width == 0 || height == 0) {
      throw std::runtime_error("resize_sensor: width and height must be > 0");
    }
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
}

void SingleCameraLoader::load_from_json(const std::string& cameras_json_path,
                                        const std::string& poses_json_path) {
  using json = nlohmann::json;

  // ---- Load cameras.json ----------------------------------------------------
  {
    std::ifstream f(cameras_json_path);
    if (!f.is_open()) {
      throw std::runtime_error("Failed to open cameras.json: " + cameras_json_path);
    }
    json cameras_json = json::parse(f);
    if (!cameras_json.is_array()) {
      throw std::runtime_error("cameras.json must be a JSON array");
    }

    m_camera_intrinsics.clear();
    m_camera_intrinsics.reserve(cameras_json.size());

    for (const auto& cam : cameras_json) {
      CameraIntrinsics intr{};
      intr.uid = cam.at("camera_id").get<uint64_t>();

      const std::string model = cam.at("model").get<std::string>();
      if (model == "PINHOLE") {
        intr.model = CameraModel::Pinhole;
      } else if (model == "SIMPLE_PINHOLE") {
        intr.model = CameraModel::Pinhole;  // treat as pinhole with fx == fy
      } else {
        throw std::runtime_error("Unsupported camera model in cameras.json: " + model);
      }

      intr.width = cam.at("width").get<int>();
      intr.height = cam.at("height").get<int>();

      const auto& params = cam.at("params");
      if (model == "SIMPLE_PINHOLE") {
        // params: [f, cx, cy]
        intr.fx = intr.fy = params.at(0).get<float>();
        intr.cx = params.at(1).get<float>();
        intr.cy = params.at(2).get<float>();
      } else {
        // PINHOLE: params: [fx, fy, cx, cy]
        intr.fx = params.at(0).get<float>();
        intr.fy = params.at(1).get<float>();
        intr.cx = params.at(2).get<float>();
        intr.cy = params.at(3).get<float>();
      }

      // No distortion from the JSON conversion pipeline
      intr.k1 = intr.k2 = intr.k3 = 0.0f;
      intr.p1 = intr.p2 = 0.0f;

      m_camera_intrinsics.push_back(intr);
    }
    log_info("Loaded {} camera intrinsics from {}", m_camera_intrinsics.size(), cameras_json_path);
  }

  // ---- Load poses.json ------------------------------------------------------
  {
    std::ifstream f(poses_json_path);
    if (!f.is_open()) {
      throw std::runtime_error("Failed to open poses.json: " + poses_json_path);
    }
    json poses_json = json::parse(f);
    if (!poses_json.is_array()) {
      throw std::runtime_error("poses.json must be a JSON array");
    }

    m_camera_extrinsics.clear();
    m_camera_extrinsics.reserve(poses_json.size());

    for (const auto& pose : poses_json) {
      const auto& qvals = pose.at("qvec");
      const auto& tvals = pose.at("tvec");

      float qw = qvals.at(0).get<float>();
      float qx = qvals.at(1).get<float>();
      float qy = qvals.at(2).get<float>();
      float qz = qvals.at(3).get<float>();
      float norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
      qw /= norm; qx /= norm; qy /= norm; qz /= norm;

      vec3 t{tvals.at(0).get<float>(), tvals.at(1).get<float>(), tvals.at(2).get<float>()};
      quat q{qw, qx, qy, qz};

      uuid_t image_id = pose.at("image_id").get<uint64_t>();
      uuid_t camera_id = pose.at("camera_id").get<uint64_t>();

      // Derive timestamp from image name (strip extension)
      std::string name = pose.at("name").get<std::string>();
      uuid_t timestamp = 0;
      // Try to extract numeric timestamp from the name
      auto dot_pos = name.rfind('.');
      std::string stem = (dot_pos != std::string::npos) ? name.substr(0, dot_pos) : name;
      try {
        timestamp = std::stoull(stem);
      } catch (...) {
        // Non-numeric name: use image_id as timestamp
        timestamp = image_id;
      }

      // Map camera_id to 0-based index in intrinsics vector
      uuid_t cam_uid = 0;
      for (size_t ci = 0; ci < m_camera_intrinsics.size(); ++ci) {
        if (m_camera_intrinsics[ci].uid == camera_id) {
          cam_uid = ci;
          break;
        }
      }

      m_camera_extrinsics.emplace_back(q, t, image_id, timestamp, cam_uid);
    }
    std::sort(m_camera_extrinsics.begin(), m_camera_extrinsics.end(),
              [](const auto& a, const auto& b) { return a.frame_idx < b.frame_idx; });
    log_info("Loaded {} camera extrinsics from {}", m_camera_extrinsics.size(), poses_json_path);
  }
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