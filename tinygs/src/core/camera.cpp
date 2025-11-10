#include <tinygs/core/camera.hpp>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace tinygs {

// Implementation of CameraIntrinsics::parse
CameraIntrinsics CameraIntrinsics::parse(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    if (tokens.size() < 13) {
        throw std::runtime_error("Invalid camera intrinsics format: expected 13 values");
    }
    
    CameraIntrinsics intrinsics;
    intrinsics.uid = std::stoull(tokens[0]);
    
    // Parse camera model
    if (tokens[1] == "PINHOLE") {
        intrinsics.model = CameraModel::Pinhole;
    } else {
        throw std::runtime_error("Unsupported camera model: " + tokens[1]);
    }
    
    intrinsics.width = std::stoi(tokens[2]);
    intrinsics.height = std::stoi(tokens[3]);
    intrinsics.fx = std::stof(tokens[4]);
    intrinsics.fy = std::stof(tokens[5]);
    intrinsics.cx = std::stof(tokens[6]);
    intrinsics.cy = std::stof(tokens[7]);
    intrinsics.k1 = std::stof(tokens[8]);
    intrinsics.k2 = std::stof(tokens[9]);
    intrinsics.k3 = std::stof(tokens[10]);
    intrinsics.p1 = std::stof(tokens[11]);
    intrinsics.p2 = std::stof(tokens[12]);
    
    return intrinsics;
}

// Implementation of CameraExtrinsics::parse
CameraExtrinsics CameraExtrinsics::parse(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    if (tokens.size() < 10) {
      throw std::runtime_error("Invalid camera extrinsics format: expected at least 10 values, got "
                               + std::to_string(tokens.size()) + "\"" + line + "\"");
    }

    // Parse camera id
    uuid_t frame_idx = std::stoull(tokens[0]);

    // Parse quaternion (qw, qx, qy, qz) and translation (tx, ty, tz)
    float qw = std::stof(tokens[1]);
    float qx = std::stof(tokens[2]);
    float qy = std::stof(tokens[3]);
    float qz = std::stof(tokens[4]);
    float tx = std::stof(tokens[5]);
    float ty = std::stof(tokens[6]);
    float tz = std::stof(tokens[7]);
    uuid_t camera_id = std::stoull(tokens[8]); // ignore.

    const auto& timestamp_token = tokens[9];
    uuid_t timestamp = 0;
    if (auto dot_position = timestamp_token.find('.');
        dot_position == std::string::npos) {
      // something like "123456", which is safe to cast directly to uuid
      timestamp = std::stoull(timestamp_token);
    } else {
      // "123456.jpg" or "34124.png", we extract the number part.
      timestamp = std::stoull(timestamp_token.substr(0, dot_position));
    }
    // Normalize quaternion
    float norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= norm; qx /= norm; qy /= norm; qz /= norm;

    return CameraExtrinsics(quat{qw, qx, qy, qz}, vec3{tx, ty, tz}, frame_idx, timestamp, camera_id);
}

std::string CameraIntrinsics::to_string() const {
  std::string model_str;
  switch (model) {
  case CameraModel::Pinhole:
    model_str = "PINHOLE";
    break;
  default:
    model_str = "UNKNOWN";
    break;
  }

  return fmt::format("CameraIntrinsics(id={}, model={}, width={}, height={}, "
                     "fx={:.3f}, fy={:.3f}, cx={:.3f}, cy={:.3f}, "
                     "k1={:.6f}, k2={:.6f}, k3={:.6f}, p1={:.6f}, p2={:.6f})",
                     uid, model_str, width, height, fx, fy, cx, cy, k1, k2, k3,
                     p1, p2);
}

std::string to_string(const CameraIntrinsics &intrinsics) {
  return intrinsics.to_string();
}
} // namespace tinygs