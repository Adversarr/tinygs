#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/cuda/vec.hpp"

namespace tinygs {

/// @brief SoA structure for 3D Gaussians
struct Gaussian3d {
  std::vector<vec3> means;
  std::vector<float> opacities;
  std::vector<vec4> rotations;
  std::vector<vec3> scales;
  std::vector<vec3> sh_coefficient_0;
  std::vector<vec3> sh_coefficients_rest;
};

struct DensificationInfo {
  float accum_counter = 0;
  float accum_grad_mean2d = 0;    /// accumulated gradient of mean2d
  float accum_absgrad_mean2d = 0; /// accumulated absolute gradient of mean2d
};

mat4x4 normalize_scene(
  const Gaussian3d& gs3d,
  const std::vector<std::pair<mat4x4, mat3x3>>& w2c_k_s,
  float ext_scale = 1.0f);

TINYGS_HOST_DEVICE inline float activate_scale(float x) {
  return ::expf(x);
  // return log(1 + exp(x));
}

TINYGS_HOST_DEVICE inline float deactivate_scale(float x) {
  return ::logf(x);
  // return log(exp(x) - 1);
}

TINYGS_HOST_DEVICE inline float activate_scale_deriv(float x) {
  return ::expf(x);
  // return logistic(x);
}

TINYGS_HOST_DEVICE inline vec3 activate_scale(const vec3& x) {
  return vec3(activate_scale(x.x), activate_scale(x.y), activate_scale(x.z));
}

TINYGS_HOST_DEVICE inline vec3 deactivate_scale(const vec3& x) {
  return vec3(deactivate_scale(x.x), deactivate_scale(x.y), deactivate_scale(x.z));
}

TINYGS_HOST_DEVICE inline vec3 activate_scale_deriv(const vec3& x) {
  return vec3(activate_scale_deriv(x.x), activate_scale_deriv(x.y), activate_scale_deriv(x.z));
}

TINYGS_HOST_DEVICE inline float activate_opacity(float x) {
  return logistic(x);
}

TINYGS_HOST_DEVICE inline float deactivate_opacity(float x) {
  return logit(x);
}

TINYGS_HOST_DEVICE inline float activate_opacity_deriv(float x) {
  const float actx = activate_opacity(x);
  return actx * (1 - actx);
}

}  // namespace tinygs