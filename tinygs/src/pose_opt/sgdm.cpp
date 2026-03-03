#include "tinygs/pose_opt/sgdm.hpp"
#include "tinygs/core/camera.hpp"
#include "axis_utils.hpp"
#include <glm/gtc/quaternion.hpp>

namespace tinygs {

PoseOptSgdM::PoseOptSgdM() = default;

mat4x4 PoseOptSgdM::query(uuid_t timestamp, mat4x4 world_to_camera) {
  auto& state = m_states[timestamp];

  // TODO: we have to check if the rotation are unchanged during optimization.
  state.m_pose = world_to_camera;

  const mat3x3 delta_rotation = pose_opt_detail::rotated_axis(state.m_axis_x_modifier, state.m_axis_y_modifier);
  const mat3x3 rot = mat3x3(state.m_pose);
  return make_w2c(delta_rotation * rot, vec3(state.m_pose[3]) + state.m_delta_translation);
}

void PoseOptSgdM::update(uuid_t timestamp, const mat4x4 &grad_pose, float step_size) {
  auto& state = m_states[timestamp];

  // Gradient w.r.t. delta rotation from overall rotation gradient
  mat3x3 total_rot_grad(grad_pose);

  const mat3x3 rot = mat3x3(state.m_pose);
  const mat3x3 dL_ddelta = total_rot_grad * glm::transpose(rot);
  
  auto [grad_ax, grad_ay] = pose_opt_detail::grad_rotated_axis(state.m_axis_x_modifier, state.m_axis_y_modifier, dL_ddelta);
  vec3 grad_tr = vec3(grad_pose[3]);

  const float lr = m_params.lr * step_size;
  const float mom = m_params.momentum;

  grad_ax += m_params.weight_decay * (state.m_axis_x_modifier - vec3(1.0f, 0.0f, 0.0f));
  grad_ay += m_params.weight_decay * (state.m_axis_y_modifier - vec3(0.0f, 1.0f, 0.0f));
  grad_tr  += m_params.weight_decay * state.m_delta_translation;

  state.m_axis_x_modifier_momentum = mom * state.m_axis_x_modifier_momentum + (1.0f - mom) * grad_ax;
  state.m_axis_y_modifier_momentum = mom * state.m_axis_y_modifier_momentum + (1.0f - mom) * grad_ay;
  state.m_delta_translation_momentum = mom * state.m_delta_translation_momentum + (1.0f - mom) * grad_tr;

  state.m_axis_x_modifier -= lr * state.m_axis_x_modifier_momentum;
  state.m_axis_y_modifier -= lr * state.m_axis_y_modifier_momentum;
  state.m_delta_translation -= lr * state.m_delta_translation_momentum;
}

json PoseOptSgdM::get_params() const noexcept {
  json ret = m_params.to_json();
  ret["type"] = "sgdm";
  return ret;
}

void PoseOptSgdM::set_params(const json &params) {
  m_params.from_json(params);
}

json PoseOptSgdM::Params::to_json() const noexcept {
  json j;
  j["lr"] = lr;
  j["momentum"] = momentum;
  j["weight_decay"] = weight_decay;
  return j;
}

void PoseOptSgdM::Params::from_json(const json &j) {
  if (j.contains("lr")) {
    lr = j["lr"].get<float>();
  }
  if (j.contains("momentum")) {
    momentum = j["momentum"].get<float>();
  }
  if (j.contains("weight_decay")) {
    weight_decay = j["weight_decay"].get<float>();
  }
}

}