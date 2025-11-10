#include "tinygs/core/camera_ext.hpp"

#include <algorithm>
#include <stdexcept>

namespace tinygs {

CameraExtrinsics interpolate(
    const CameraExtrinsics& a,
    const CameraExtrinsics& b,
    float t,
    InterpolationMethod method) {
  t = std::clamp(t, 0.0f, 1.0f);

  switch (method) {
    case InterpolationMethod::Linear: {
      // Normalize quaternions and ensure shortest-path interpolation
      quat qa = glm::normalize(a.m_q);
      quat qb = glm::normalize(b.m_q);
      if (glm::dot(qa, qb) < 0.0f) {
        qb = -qb;
      }

      quat q = glm::slerp(qa, qb, t);
      vec3 tr = a.m_t * (1.0f - t) + b.m_t * t;
      return CameraExtrinsics(q, tr, a.frame_idx, a.timestamp, a.cam_uid);
    }
    default:
      throw std::runtime_error("Unsupported interpolation method");
  }
}

} // namespace tinygs