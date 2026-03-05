#pragma once
#include <tinygs/common.hpp>
#include <tinygs/math/vec.hpp>
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
  uuid_t uid = 0; ///< Camera ID (0-based)
  CameraModel model = CameraModel::Pinhole;
  int width = 0;
  int height = 0;
  float fx = 0.0f, fy = 0.0f;  ///< Focal lengths
  float cx = 0.0f, cy = 0.0f;  ///< Principal point
  float k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;  ///< Radial distortion coefficients
  float p1 = 0.0f, p2 = 0.0f;      ///< Tangential distortion coefficients
  
  /// @brief Get 3x3 intrinsics matrix K
  TINYGS_HOST_DEVICE mat3x3 to_mat3() const {
    return mat3x3{
      fx, 0.0f, 0.f, // col 0
      0.0f, fy, 0.f, // col 1
      cx,   cy, 1.0f // col 2
    };
  }
  
  /// @brief Parse camera intrinsics from COLMAP format string
  static CameraIntrinsics parse(const std::string& line);
  
  /// @brief Convert camera intrinsics to string representation
  std::string to_string() const;
};

inline mat4x4 make_w2c(const mat3x3& rot, const vec3& t) {
  mat4x4 w2c = rot;
  w2c[3][0] = t[0];
  w2c[3][1] = t[1];
  w2c[3][2] = t[2];
  return w2c;
}

inline mat4x4 make_w2c(const quat& q, const vec3& t) {
  return make_w2c(quat_to_mat3(q), t);
}

/// @brief Global to_string function for CameraIntrinsics
std::string to_string(const CameraIntrinsics &intrinsics);

/// @brief Camera extrinsic parameters (pose)
struct CameraExtrinsics {
  quat m_q;            ///< Quaternion (qw, qx, qy, qz)
  vec3 m_t;            ///< Translation (tx, ty, tz)
  uuid_t frame_idx;    ///< Frame IDX (1-based)
  uuid_t timestamp;    ///< Global timestamp
  uuid_t cam_uid;      ///< Camera ID (0-based)
  
  /// @brief Constructor from quaternion and translation
  TINYGS_HOST_DEVICE CameraExtrinsics(const quat& quaternion, const vec3& translation,  //
                                      uuid_t frame_uid, uuid_t timestamp, uuid_t cam_uid) :
      m_q(quaternion), m_t(translation), frame_idx(frame_uid), timestamp(timestamp), cam_uid(cam_uid) {}

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
    return vec3{c2w[3][0], c2w[3][1], c2w[3][2]};
  }
  
  /// @brief Get camera forward direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_forward() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return -vec3{c2w[2][0], c2w[2][1], c2w[2][2]};
  }
  
  /// @brief Get camera up direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_up() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[1][0], c2w[1][1], c2w[1][2]};
  }
  
  /// @brief Get camera right direction in world coordinates
  TINYGS_HOST_DEVICE vec3 get_right() const {
    mat4x4 c2w = extrinsics.get_c2w();
    return vec3{c2w[0][0], c2w[0][1], c2w[0][2]};
  }
};

}
