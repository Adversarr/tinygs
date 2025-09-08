#pragma once
#include <tinygs/cuda/common_host.hpp>
#include <tinygs/cuda/vec.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace tinygs {

enum class CameraModel: int {
  Pinhole,
  MaxCameraModel,
};

// Camera intrinsic parameters (corresponds to Python Camera dataclass)
struct CameraIntrinsics {
  int id;
  CameraModel model;
  int width;
  int height;
  float fx, fy;  // focal lengths
  float cx, cy;  // principal point
  float k1, k2, k3;  // radial distortion coefficients
  float p1, p2;      // tangential distortion coefficients
  
  // Get 3x3 intrinsics matrix K
  TINYGS_HOST_DEVICE mat3x3 get_K() const {
    return mat3x3{
      fx, 0.0f, 0.f,
      0.0f, fy, 0.f,
      cx,   cy, 1.0f
    };
  }
  
  // Parse camera intrinsics from COLMAP format string
  // Format: "CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY K1 K2 K3 P1 P2"
  static CameraIntrinsics parse(const std::string& line);
  
  // Convert camera intrinsics to string representation
  std::string to_string() const {
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
                      id, model_str, width, height, fx, fy, cx, cy, k1, k2, k3, p1, p2);
  }
};

// Global to_string function for CameraIntrinsics
inline std::string to_string(const CameraIntrinsics& intrinsics) {
  return intrinsics.to_string();
}

// Camera extrinsic parameters (pose)
struct CameraExtrinsics {
  quat m_q;  // quaternion (qw, qx, qy, qz)
  vec3 m_t;  // translation (tx, ty, tz)
  /// @brief The camera id of the data.
  uint32_t frame_uid;

  // Constructor from quaternion and translation
  TINYGS_HOST_DEVICE CameraExtrinsics(const quat& quaternion, const vec3& translation, uint32_t frame_uid) 
    : m_q(quaternion), m_t(translation), frame_uid(frame_uid) {}

  // Default constructor
  CameraExtrinsics() = default;

  // Get world-to-camera transformation matrix
  TINYGS_HOST_DEVICE mat4x4 get_w2c() const {
    mat4x4 w2c{quat_to_mat3(m_q)};
    w2c[3][0] = m_t[0];
    w2c[3][1] = m_t[1];
    w2c[3][2] = m_t[2];
    return w2c;
  }

  // Get camera-to-world transformation matrix
  TINYGS_HOST_DEVICE mat4x4 get_c2w() const {
    return inverse(get_w2c());
  }
  
  // Parse camera extrinsics from trajectory format string
  // Format: "id qw qx qy qz tx ty tz ... (ignore the rest)"
  static CameraExtrinsics parse(const std::string& line);
};

class Camera final {
public:
  CameraIntrinsics intrinsics;
  CameraExtrinsics extrinsics;
  
  // Constructor
  TINYGS_HOST_DEVICE Camera(const CameraIntrinsics& intr, const CameraExtrinsics& extr)
    : intrinsics(intr), extrinsics(extr) {}
  
  // Default constructor
  Camera() = default;

  // Get camera position in world coordinates
  TINYGS_HOST_DEVICE vec3 get_position() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[0][3], c2w[1][3], c2w[2][3]};
  }
  
  // Get camera forward direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_forward() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return -vec3{c2w[0][2], c2w[1][2], c2w[2][2]};
  }
  
  // Get camera up direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_up() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[0][1], c2w[1][1], c2w[2][1]};
  }
  
  // Get camera right direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_right() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[0][0], c2w[1][0], c2w[2][0]};
  }
};

}