#include "tinygs/pose_opt/adamw.hpp"
#include "tinygs/core/camera.hpp"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace tinygs {

// Build an orthonormal rotation from two (approximately) axis vectors
static inline mat3x3 rotated_axis(vec3 x_axis, vec3 y_axis) {
  const vec3 x = glm::normalize(x_axis);
  const vec3 y = glm::normalize(y_axis - glm::dot(y_axis, x) * x);
  return mat3x3(x, y, glm::cross(x, y));
}

static inline mat3x3 outer_product(const vec3& a, const vec3& b) {
  // Column-major: columns are a*b.x, a*b.y, a*b.z
  return mat3x3(a * b.x, a * b.y, a * b.z);
}

// Gradient of rotated_axis w.r.t. the x/y axis modifiers
static inline std::pair<vec3, vec3> grad_rotated_axis(const vec3& x_axis, const vec3& y_axis, const mat3x3& dL_drot) {
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

mat4x4 PoseOptAdamW::query(uuid_t timestamp, mat4x4 world_to_camera) {
  auto& state = m_states[timestamp];

  // Keep the original pose for rotation baseline
  state.m_pose = world_to_camera;

  const mat3x3 delta_rotation = rotated_axis(state.m_axis_x_modifier, state.m_axis_y_modifier);
  const mat3x3 rot = mat3x3(state.m_pose);
  return make_w2c(delta_rotation * rot, vec3(state.m_pose[3]) + state.m_delta_translation);
}

void PoseOptAdamW::update(uuid_t timestamp, const mat4x4 &grad_pose, float step_size) {
  auto& state = m_states[timestamp];

  // Gradient w.r.t. delta rotation from overall rotation gradient
  const mat3x3 total_rot_grad(grad_pose);
  const mat3x3 rot = mat3x3(state.m_pose);
  const mat3x3 dL_ddelta = total_rot_grad * glm::transpose(rot);

  auto [grad_ax, grad_ay] = grad_rotated_axis(state.m_axis_x_modifier, state.m_axis_y_modifier, dL_ddelta);
  vec3 grad_tr = vec3(grad_pose[3]);

  // AdamW parameters
  const float lr = m_params.lr * step_size;
  const float b1 = m_params.beta1;
  const float b2 = m_params.beta2;
  const float eps = m_params.epsilon;
  const float wd = m_params.weight_decay;

  state.m_steps += 1;
  const float b1_corr = 1.0f - std::pow(b1, (float)state.m_steps);
  const float b2_corr = 1.0f - std::pow(b2, (float)state.m_steps);

  // First moment updates
  state.m_axis_x_m = b1 * state.m_axis_x_m + (1.0f - b1) * grad_ax;
  state.m_axis_y_m = b1 * state.m_axis_y_m + (1.0f - b1) * grad_ay;
  state.m_delta_translation_m = b1 * state.m_delta_translation_m + (1.0f - b1) * grad_tr;

  // Second moment updates (element-wise square)
  state.m_axis_x_v = b2 * state.m_axis_x_v + (1.0f - b2) * (grad_ax * grad_ax);
  state.m_axis_y_v = b2 * state.m_axis_y_v + (1.0f - b2) * (grad_ay * grad_ay);
  state.m_delta_translation_v = b2 * state.m_delta_translation_v + (1.0f - b2) * (grad_tr * grad_tr);

  // Bias correction
  vec3 mhat_ax = state.m_axis_x_m / b1_corr;
  vec3 vhat_ax = state.m_axis_x_v / b2_corr;
  vec3 mhat_ay = state.m_axis_y_m / b1_corr;
  vec3 vhat_ay = state.m_axis_y_v / b2_corr;
  vec3 mhat_tr = state.m_delta_translation_m / b1_corr;
  vec3 vhat_tr = state.m_delta_translation_v / b2_corr;

  // Parameter updates
  const vec3 epsv(eps);
  vec3 upd_ax = lr * (mhat_ax / (glm::sqrt(vhat_ax) + epsv));
  vec3 upd_ay = lr * (mhat_ay / (glm::sqrt(vhat_ay) + epsv));
  vec3 upd_tr = lr * (mhat_tr / (glm::sqrt(vhat_tr) + epsv));

  // Decoupled weight decay (anchor axis modifiers to canonical axes)
  const vec3 ax_base(1.0f, 0.0f, 0.0f);
  const vec3 ay_base(0.0f, 1.0f, 0.0f);

  state.m_axis_x_modifier -= upd_ax + lr * wd * (state.m_axis_x_modifier - ax_base);
  state.m_axis_y_modifier -= upd_ay + lr * wd * (state.m_axis_y_modifier - ay_base);
  state.m_delta_translation -= upd_tr + lr * wd * state.m_delta_translation;
}

json PoseOptAdamW::get_params() const noexcept {
  json ret = m_params.to_json();
  ret["type"] = "adamw";
  return ret;
}

void PoseOptAdamW::set_params(const json &params) {
  m_params.from_json(params);
}

void PoseOptAdamW::Params::from_json(const json &j) {
  if (j.contains("lr"))
    lr = j["lr"].get<float>();
  if (j.contains("beta1"))
    beta1 = j["beta1"].get<float>();
  if (j.contains("beta2"))
    beta2 = j["beta2"].get<float>();
  if (j.contains("epsilon"))
    epsilon = j["epsilon"].get<float>();
  if (j.contains("weight_decay"))
    weight_decay = j["weight_decay"].get<float>();
}

json PoseOptAdamW::Params::to_json() const noexcept {
  json j;
  j["lr"] = lr;
  j["beta1"] = beta1;
  j["beta2"] = beta2;
  j["epsilon"] = epsilon;
  j["weight_decay"] = weight_decay;
  return j;
}

} // namespace tinygs