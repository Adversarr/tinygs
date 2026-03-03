#pragma once

#include "tinygs/core/camera.hpp"

#include <algorithm>
#include <utility>

namespace tinygs::pose_opt_detail {

inline mat3x3 rotated_axis(vec3 x_axis, vec3 y_axis) {
  const vec3 x = glm::normalize(x_axis);
  const vec3 y = glm::normalize(y_axis - glm::dot(y_axis, x) * x);
  return mat3x3(x, y, glm::cross(x, y));
}

inline mat3x3 outer_product(const vec3& a, const vec3& b) {
  return mat3x3(a * b.x, a * b.y, a * b.z);
}

inline std::pair<vec3, vec3> grad_rotated_axis(const vec3& x_axis, const vec3& y_axis, const mat3x3& dL_drot) {
  const float eps = 1e-8f;

  const vec3 a = x_axis;
  const vec3 v = y_axis;

  const float a_norm = std::max(glm::length(a), eps);
  const vec3 x = a / a_norm;

  const float vx = glm::dot(v, x);
  const vec3 y_prime = v - vx * x;
  const float yprime_norm = std::max(glm::length(y_prime), eps);
  const vec3 y = y_prime / yprime_norm;

  const vec3 gz = dL_drot[2];
  const vec3 gy = dL_drot[1];
  const vec3 gx = dL_drot[0];

  vec3 g_x = gx + glm::cross(gz, y);
  vec3 g_y = gy + glm::cross(x, gz);

  const mat3x3 I(1.0f);
  mat3x3 J_norm_y = I - outer_product(y, y);
  J_norm_y /= yprime_norm;
  const vec3 g_yprime = J_norm_y * g_y;

  const mat3x3 dv = I - outer_product(x, x);
  vec3 grad_v = dv * g_yprime;

  const mat3x3 M = outer_product(v, x) + vx * I;
  g_x += -(M * g_yprime);

  mat3x3 J_norm_x = I - outer_product(x, x);
  J_norm_x /= a_norm;
  const vec3 grad_a = J_norm_x * g_x;

  return {grad_a, grad_v};
}

}  // namespace tinygs::pose_opt_detail
