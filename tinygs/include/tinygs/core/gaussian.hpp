#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/cuda/vec.hpp"

namespace tinygs {

/**
 * @brief 2D Gaussian both Host and GPU memory, it is small, efficient enough that SoA does not work better.
 * @description It aligns to 16 bytes for better memory access performance.
 * @todo this structure have 36 Bytes in memory, which is not a good alignment
 */
struct Gaussian2dItem {
  // mean2d
  vec2 mean;
  // color
  vec3 rgb;
  // conic matrix(2x2, sym, store 3) and opacity
  vec4 conic_opacity;
  // other auxiliary variables...
};

/**
 * @brief SoA structure for 3D Gaussians
 * 
 */
struct Gaussian3d {
  std::vector<vec3> means;
  std::vector<float> opacities;
  std::vector<vec4> rotations;
  std::vector<vec3> scales;
  std::vector<vec3> sh_coefficient_0;
  std::vector<vec3> sh_coefficients_rest;
};

mat4x4 normalize_scene(
  const Gaussian3d& gs3d,
  const std::vector<std::pair<mat4x4, mat3x3>>& w2c_k_s,
  float ext_scale = 1.0f
);



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