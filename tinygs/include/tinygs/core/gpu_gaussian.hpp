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
  const thrust::device_vector<vec3>& sh_coefficient_0() const { return m_sh_coefficient_0; }
  const thrust::device_vector<vec3>& sh_coefficients_rest() const { return m_sh_coefficients_rest; }

  thrust::device_vector<vec3>& means() { return m_means; }
  thrust::device_vector<float>& opacities() { return m_opacities; }
  thrust::device_vector<vec4>& rotations() { return m_rotations; }
  thrust::device_vector<vec3>& scales() { return m_scales; }
  thrust::device_vector<vec3>& sh_coefficient_0() { return m_sh_coefficient_0; }
  thrust::device_vector<vec3>& sh_coefficients_rest() { return m_sh_coefficients_rest; }

  std::unique_ptr<GPUGaussian3d> clone_async(cudaStream_t stream = 0);
  std::unique_ptr<GPUGaussian3d> clone();

  void memset_async(char value, cudaStream_t stream);
  void memset(char value);

  void remove(char* kept_flag, int num_kept);

  void append(int num_dup);

  float scene_scale() const { return m_scene_scale; }

  void set_scene_scale(float scale) { m_scene_scale = scale; }

  void set_sh_degree(int degree) { m_current_sh_degree = std::min(kMaxSphericalHarmonicsDegree, degree); }

  int get_sh_degree() const { return m_current_sh_degree; }

private:
  /// NOTE: Change to gaussians is not frequent, thrust generally have a good performance
  /// NOTE: SoA of Gaussian3d in GPU memory

  /// TODO: many metrics indicates that, AoS structure may be more efficient than SoA in this case.

  int m_current_sh_degree = 0;
  float m_scene_scale = 1.0f;

  // means
  thrust::device_vector<vec3> m_means;
  // opacities
  thrust::device_vector<float> m_opacities;
  // rotations are packed in a single vec4
  thrust::device_vector<vec4> m_rotations;
  // scales are packed in a single vec3
  thrust::device_vector<vec3> m_scales;
  // sh coefficients are packed in a single vec3, 16 coefficients per color.
  // the sh0 is special.
  thrust::device_vector<vec3> m_sh_coefficient_0;
  thrust::device_vector<vec3> m_sh_coefficients_rest;
};

}  // namespace tinygs