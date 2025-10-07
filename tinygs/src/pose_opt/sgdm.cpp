#include "tinygs/pose_opt/sgdm.hpp"
#include "tinygs/core/camera.hpp"
#include <glm/gtc/quaternion.hpp>

namespace tinygs {

PoseOptSgdM::PoseOptSgdM() = default;

inline mat3x3 rotated_axis(vec3 x_axis, vec3 y_axis) {
  const vec3 x = glm::normalize(x_axis);
  const vec3 y = glm::normalize(y_axis - glm::dot(y_axis, x) * x);
  return mat3x3(x, y, glm::cross(x, y));
}

inline mat3x3 outer_product(const vec3& a, const vec3& b) {
  // Column-major: columns are a*b.x, a*b.y, a*b.z
  return mat3x3(a * b.x, a * b.y, a * b.z);
}

inline std::pair<vec3, vec3> grad_rotated_axis(const vec3& x_axis, const vec3& y_axis, const mat3x3& dL_drot) {
  const float eps = 1e-8f;

  // Forward pass to compute orthonormal axes from modifiers
  const vec3 a = x_axis;
  const vec3 v = y_axis;

  const float a_norm = std::max(glm::length(a), eps);
  const vec3 x = a / a_norm;

  const float vx = glm::dot(v, x);
  const vec3 y_prime = v - vx * x;
  const float yprime_norm = std::max(glm::length(y_prime), eps);
  const vec3 y = y_prime / yprime_norm;

  // z = x × y; gradients from z propagate to x and y
  const vec3 gz = dL_drot[2];
  const vec3 gy = dL_drot[1];
  const vec3 gx = dL_drot[0];

  vec3 g_x = gx + glm::cross(gz, y);
  vec3 g_y = gy + glm::cross(x, gz);

  // Backprop through y = normalize(y')
  const mat3x3 I(1.0f);
  mat3x3 J_norm_y = I - outer_product(y, y);
  J_norm_y /= yprime_norm;
  const vec3 g_yprime = J_norm_y * g_y;

  // Backprop through y' = v - (v·x) x
  const mat3x3 dv = I - outer_product(x, x);
  vec3 grad_v = dv * g_yprime;

  const mat3x3 M = outer_product(v, x) + vx * I; // dy'/dx = -(v x^T + (v·x) I)
  g_x += -(M * g_yprime);

  // Backprop through x = normalize(a)
  mat3x3 J_norm_x = I - outer_product(x, x);
  J_norm_x /= a_norm;
  const vec3 grad_a = J_norm_x * g_x;

  return {grad_a, grad_v};
}

mat4x4 PoseOptSgdM::query(uuid_t timestamp, mat4x4 world_to_camera) {
  auto& state = m_states[timestamp];

  // TODO: we have to check if the rotation are unchanged during optimization.
  state.m_pose = world_to_camera;

  const mat3x3 delta_rotation = rotated_axis(state.m_axis_x_modifier, state.m_axis_y_modifier);
  const mat3x3 rot = mat3x3(state.m_pose);
  return make_w2c(delta_rotation * rot, vec3(state.m_pose[3]) + state.m_delta_translation);
}

void PoseOptSgdM::update(uuid_t timestamp, const mat4x4 &grad_pose, float step_size) {
  auto& state = m_states[timestamp];

  // Gradient w.r.t. delta rotation from overall rotation gradient
  mat3x3 total_rot_grad(grad_pose);

  const mat3x3 rot = mat3x3(state.m_pose);
  const mat3x3 dL_ddelta = total_rot_grad * glm::transpose(rot);
  
  auto [grad_ax, grad_ay] = grad_rotated_axis(state.m_axis_x_modifier, state.m_axis_y_modifier, dL_ddelta);
  vec3 grad_tr = vec3(grad_pose[3]);
  
  // const float dL_ddelta_norm = glm::length(dL_ddelta[0]) + glm::length(dL_ddelta[1]) + glm::length(dL_ddelta[2]);
  // const float dL_dtranslation = glm::length(grad_t);
  // log_info("dL_ddelta_norm: {}, dL_dtranslation: {}", dL_ddelta_norm, dL_dtranslation);

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