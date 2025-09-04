#pragma once
#include "tinygs/common.hpp"
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

}  // namespace tinygs