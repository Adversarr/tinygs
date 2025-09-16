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

/// @brief Camera intrinsic parameters
struct CameraIntrinsics {
  int uid; ///< Camera ID (0-based)
  CameraModel model;
  int width;
  int height;
  float fx, fy;  ///< Focal lengths
  float cx, cy;  ///< Principal point
  float k1, k2, k3;  ///< Radial distortion coefficients
  float p1, p2;      ///< Tangential distortion coefficients
  
  /// @brief Get 3x3 intrinsics matrix K
  TINYGS_HOST_DEVICE mat3x3 to_mat3() const {
    return mat3x3{
      fx, 0.0f, 0.f,
      0.0f, fy, 0.f,
      cx,   cy, 1.0f
    };
  }
  
  /// @brief Parse camera intrinsics from COLMAP format string
  static CameraIntrinsics parse(const std::string& line);
  
  /// @brief Convert camera intrinsics to string representation
  std::string to_string() const;
};

/// @brief Global to_string function for CameraIntrinsics
std::string to_string(const CameraIntrinsics &intrinsics);

/// @brief Camera extrinsic parameters (pose)
struct CameraExtrinsics {
  quat m_q;  ///< Quaternion (qw, qx, qy, qz)
  vec3 m_t;  ///< Translation (tx, ty, tz)
  uint32_t frame_uid; ///< Frame ID (1-based)

  /// @brief Constructor from quaternion and translation
  TINYGS_HOST_DEVICE CameraExtrinsics(const quat& quaternion, const vec3& translation, uint32_t frame_uid) 
    : m_q(quaternion), m_t(translation), frame_uid(frame_uid) {}

  /// @brief Default constructor
  CameraExtrinsics() = default;

  /// @brief Get world-to-camera transformation matrix
  TINYGS_HOST_DEVICE mat4x4 get_w2c() const {
    mat4x4 w2c{quat_to_mat3(m_q)};
    w2c[3][0] = m_t[0];
    w2c[3][1] = m_t[1];
    w2c[3][2] = m_t[2];
    return w2c;
  }

  /// @brief Get camera-to-world transformation matrix
  TINYGS_HOST_DEVICE mat4x4 get_c2w() const {
    return inverse(get_w2c());
  }
  
  /// @brief Parse camera extrinsics from trajectory format string
  static CameraExtrinsics parse(const std::string& line);
};

class Camera final {
public:
  CameraIntrinsics intrinsics;
  CameraExtrinsics extrinsics;
  
  /// @brief Constructor
  TINYGS_HOST_DEVICE Camera(const CameraIntrinsics& intr, const CameraExtrinsics& extr)
    : intrinsics(intr), extrinsics(extr) {}
  
  /// @brief Default constructor
  Camera() = default;

  /// @brief Get camera position in world coordinates
  TINYGS_HOST_DEVICE vec3 get_position() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[0][3], c2w[1][3], c2w[2][3]};
  }
  
  /// @brief Get camera forward direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_forward() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return -vec3{c2w[0][2], c2w[1][2], c2w[2][2]};
  }
  
  /// @brief Get camera up direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_up() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[0][1], c2w[1][1], c2w[2][1]};
  }
  
  /// @brief Get camera right direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_right() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[0][0], c2w[1][0], c2w[2][0]};
  }
};

}