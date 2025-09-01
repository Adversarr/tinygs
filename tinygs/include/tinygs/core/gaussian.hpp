#pragma once
#include "tinygs/common.hpp"
#include "tinygs/cuda/vec.hpp"

namespace tinygs {

/**
 * @brief 3D Gaussian
 * @description It aligns to 16 bytes for better memory access performance.
 * @note For kMaxSphericalHarmonicsDegree = 3, we have 240 Bytes per Gaussian
 */
struct Gaussian3d {
  // [mean3D, opacity]
  alignas(16) vec4 mean_opacity;
  // quaternion rotation
  alignas(16) vec4 rotation;
  // scale3D, pad 4B
  alignas(16) vec3 scale;
  // Spherical Harmonics coefficients
  alignas(16) vec3 sh_coefficients[kMaxSphericalHarmonicsCoefficients];
};

/**
 * @brief 2D Gaussian
 * @description It aligns to 16 bytes for better memory access performance.
 * @note For kMaxSphericalHarmonicsDegree = 3, we have 120 Bytes per Gaussian
 *
 * @todo this structure have 36 Bytes in memory, which is not a good alignment
 */
struct Gaussian2d {
  // mean2d
  vec2 mean;
  // color
  vec3 rgb;
  // conic matrix(2x2, sym, store 3) and opacity
  vec4 conic_opacity;
  // other auxiliary variables...
};

TINYGS_HOST_DEVICE void project_2d(Gaussian2d& out, const Gaussian3d& in);

}  // namespace tinygs