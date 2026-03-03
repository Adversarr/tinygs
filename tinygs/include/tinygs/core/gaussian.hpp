#pragma once
#include "tinygs/common.hpp"
#include "tinygs/core/camera.hpp"
#include "tinygs/cuda/vec.hpp"

namespace tinygs {

/// @brief Number of SH coefficients per degree: degree 0 has 1, degree 1 has 3, degree 2 has 5,
///        degree 3 has 7. Total across all degrees = (max_degree+1)^2 = 16.
constexpr int kSHDegreeNumCoeffs[] = {1, 3, 5, 7};

/// @brief SoA structure for 3D Gaussians (CPU-side, AoS for SH coefficients).
///
/// SH coefficients are stored per-degree matching the KHR_gaussian_splatting standard:
///   - sh0: 1 coefficient per Gaussian (degree 0, DC term, RGB color)
///   - sh1: 3 coefficients per Gaussian (degree 1)
///   - sh2: 5 coefficients per Gaussian (degree 2)
///   - sh3: 7 coefficients per Gaussian (degree 3)
///
/// Each coefficient is a vec3 (RGB). CPU-side layout is AoS per Gaussian:
///   sh1 = [G0_c0, G0_c1, G0_c2, G1_c0, G1_c1, G1_c2, ...]
struct Gaussian3d {
  std::vector<vec3> means;
  std::vector<float> opacities;
  std::vector<vec4> rotations;
  std::vector<vec3> scales;
  std::vector<vec3> sh0;  ///< Degree 0: 1 coeff/Gaussian (DC term)
  std::vector<vec3> sh1;  ///< Degree 1: 3 coeffs/Gaussian
  std::vector<vec3> sh2;  ///< Degree 2: 5 coeffs/Gaussian
  std::vector<vec3> sh3;  ///< Degree 3: 7 coeffs/Gaussian
};

/// @brief Per-Gaussian densification statistics accumulated during training.
///
/// Populated by the rasterizer backward pass (accum_counter, accum_grad_mean2d,
/// accum_absgrad_mean2d, max_radii_screen) and optionally by the FastGS strategy
/// (metric_importance_score, metric_pruning_score).
struct DensificationInfo {
  float accum_counter = 0;           ///< Number of views that touched this Gaussian
  float accum_grad_mean2d = 0;       ///< Accumulated signed gradient of mean2d norm
  float accum_absgrad_mean2d = 0;    ///< Accumulated absolute gradient of mean2d norm
  float max_radii_screen = 0;        ///< Maximum radius in screen space
  float metric_importance_score = 0; ///< FastGS: multi-view importance score
  float metric_pruning_score = 0;    ///< FastGS: reconstruction quality pruning score
};

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