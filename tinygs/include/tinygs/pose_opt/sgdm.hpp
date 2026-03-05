#pragma once
#include "tinygs/pose_opt/pose_opt.hpp"
#include <unordered_map>

namespace tinygs {

class PoseOptSgdM: public PoseOptBase {
public:
  PoseOptSgdM();
  virtual ~PoseOptSgdM() = default;

  mat4x4 query(uuid_t timestamp, mat4x4 world_to_camera) override;

  void update(uuid_t timestamp, const mat4x4& grad_pose, float step_size) override;

  json get_params() const noexcept override;

  void set_params(const json& params) override;

  struct Params {
    float lr = 0.01f;
    float momentum = 0.9f;
    float weight_decay = 0.01f;

    json to_json() const noexcept;
    void from_json(const json& j);
  };

  struct State {
    mat4x4 m_pose = mat4x4(0.0f);

    vec3 m_axis_x_modifier = vec3(1.0f, 0.0f, 0.0f);
    vec3 m_axis_y_modifier = vec3(0.0f, 1.0f, 0.0f);
    vec3 m_delta_translation = vec3(0.0f, 0.0f, 0.0f);

    vec3 m_axis_x_modifier_momentum = vec3(0.0f, 0.0f, 0.0f);
    vec3 m_axis_y_modifier_momentum = vec3(0.0f, 0.0f, 0.0f);
    vec3 m_delta_translation_momentum = vec3(0.0f, 0.0f, 0.0f);
  };

private:
  Params m_params;

  std::unordered_map<uuid_t, State> m_states;
};

}  // namespace tinygs
