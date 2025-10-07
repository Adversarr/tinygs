#pragma once
#include "tinygs/core/camera.hpp"
#include "tinygs/pose_opt/pose_opt.hpp"

namespace tinygs {

/// @brief Pose optimizer that does not perform any optimization
class PoseOptNone : public PoseOptBase {
public:
  PoseOptNone() = default;
  virtual ~PoseOptNone() = default;
  
  mat4x4 query(uuid_t /* timestamp */, mat4x4 world_to_camera) override {
    return world_to_camera;
  }

  void update(uuid_t /* timestamp */, const mat4x4& /* grad_pose */, float /* step_size */) override {
    // No update logic
  }

  json get_params() const noexcept override {
    auto ret = json::object();
    ret["type"] = "none";
    return ret;
  }

  void set_params(const json& /* params */) override {
    // No set_params logic
  }
};

}  // namespace tinygs