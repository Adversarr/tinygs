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

  size_t size() const { return m_means.size(); }

  const thrust::device_vector<vec3>& means() const { return m_means; }
  const thrust::device_vector<float>& opacities() const { return m_opacities; }
  const thrust::device_vector<vec4>& rotations() const { return m_rotations; }
  const thrust::device_vector<vec3>& scales() const { return m_scales; }
  const thrust::device_vector<vec3>& sh_coefficients() const { return m_sh_coefficients; }
  const thrust::device_vector<vec2>& densification_info() const { return m_densification_info; }

  thrust::device_vector<vec3>& means() { return m_means; }
  thrust::device_vector<float>& opacities() { return m_opacities; }
  thrust::device_vector<vec4>& rotations() { return m_rotations; }
  thrust::device_vector<vec3>& scales() { return m_scales; }
  thrust::device_vector<vec3>& sh_coefficients() { return m_sh_coefficients; }
  thrust::device_vector<vec2>& densification_info() { return m_densification_info; }

private:
  /// NOTE: Change to gaussians is not frequent, thrust generally have a good performance
  /// NOTE: SoA of Gaussian3d in GPU memory

  /// TODO: many metrics indicates that, AoS structure may be more efficient than SoA in this case.

  int m_current_sh_degree = 0;
  float m_scene_scale = 0.0f;

  // screen space gradient to indicate which gaussian to densify
  thrust::device_vector<vec2> m_densification_info;
  // means
  thrust::device_vector<vec3> m_means;
  // opacities
  thrust::device_vector<float> m_opacities;
  // rotations are packed in a single vec4
  thrust::device_vector<vec4> m_rotations;
  // scales are packed in a single vec3
  thrust::device_vector<vec3> m_scales;
  // sh coefficients are packed in a single vec3, 16 coefficients per color.
  thrust::device_vector<vec3> m_sh_coefficients;
};

}  // namespace tinygs