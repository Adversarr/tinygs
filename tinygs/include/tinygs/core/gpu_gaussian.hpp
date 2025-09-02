#pragma once

#include "tinygs/core/gaussian.hpp"

#include <thrust/device_vector.h>

namespace tinygs {

class GPUGaussian3d {
public:
  GPUGaussian3d() = default;

  ~GPUGaussian3d() = default;

  void copy_from_host(const Gaussian3d &gaussians);

  void copy_to_host(Gaussian3d& gaussians);

  size_t size() const { return m_means_opacities.size(); }

  const thrust::device_vector<vec4>& means_opacities() const { return m_means_opacities; }
  const thrust::device_vector<vec4>& rotations() const { return m_rotations; }
  const thrust::device_vector<vec3>& scales() const { return m_scales; }
  const thrust::device_vector<vec3>& sh_coefficients() const { return m_sh_coefficients; }

private:
  /// NOTE: Change to gaussians is not frequent, thrust generally have a good performance
  /// NOTE: SoA of Gaussian3d in GPU memory

  int m_current_sh_degree = 0;
  float m_scene_scale = 0.0f;

  // means and opacities are packed in a single vec4
  thrust::device_vector<vec4> m_means_opacities;
  // rotations are packed in a single vec4
  thrust::device_vector<vec4> m_rotations;
  // scales are packed in a single vec3
  thrust::device_vector<vec3> m_scales;
  // sh coefficients are packed in a single vec3
  thrust::device_vector<vec3> m_sh_coefficients;
};

}  // namespace tinygs