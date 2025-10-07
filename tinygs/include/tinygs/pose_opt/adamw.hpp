#pragma once
#include "tinygs/cuda/vec.hpp"
#include "tinygs/common.hpp"
#include "tinygs/pose_opt/pose_opt.hpp"

namespace tinygs {

class PoseOptAdamW : public PoseOptBase {
public:
  PoseOptAdamW() = default;
  virtual ~PoseOptAdamW() = default;

  mat4x4 query(uuid_t timestamp, mat4x4 world_to_camera) override;

  void update(uuid_t timestamp, const mat4x4& grad_pose, float step_size) override;

  json get_params() const noexcept override;

  void set_params(const json& params) override;

  struct Params {
    float lr = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.01f; // decoupled

    json to_json() const noexcept;
    void from_json(const json &j);
  };

  struct State {
    mat4x4 m_pose = mat4x4(0.0f);

    vec3 m_axis_x_modifier = vec3(1.0f, 0.0f, 0.0f);
    vec3 m_axis_y_modifier = vec3(0.0f, 1.0f, 0.0f);
    vec3 m_delta_translation = vec3(0.0f, 0.0f, 0.0f);

    // Adam first and second moments
    vec3 m_axis_x_m = vec3(0.0f);
    vec3 m_axis_x_v = vec3(0.0f);
    vec3 m_axis_y_m = vec3(0.0f);
    vec3 m_axis_y_v = vec3(0.0f);
    vec3 m_delta_translation_m = vec3(0.0f);
    vec3 m_delta_translation_v = vec3(0.0f);

    uint32_t m_steps = 0;
  };

private:
  Params m_params;

  std::unordered_map<uuid_t, State> m_states;
};

}  // namespace tinygs