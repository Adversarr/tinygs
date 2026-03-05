#pragma once

#include "tinygs/core/gaussian.hpp"
#include "tinygs/platform/backend_types.hpp"

#include <thrust/device_vector.h>

namespace tinygs {

/// @brief GPU-side Gaussian data with SoA layout.
///
/// SH coefficients use a channel-first SoA layout on GPU for maximum memory throughput:
///   For degree d with C coefficients per Gaussian, the layout is:
///     [c0_R_0..c0_R_{N-1}, c0_G_0..c0_G_{N-1}, c0_B_0..c0_B_{N-1},
///      c1_R_0..c1_R_{N-1}, c1_G_0..c1_G_{N-1}, c1_B_0..c1_B_{N-1}, ...]
///   So for coefficient k, channel c (0=R,1=G,2=B), Gaussian i:
///     index = (k * 3 + c) * N + i
///
/// CPU-side Gaussian3d uses AoS (vec3 per coefficient). Conversion is done in
/// copy_from_host() / copy_to_host().
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

  thrust::device_vector<vec3>& means() { return m_means; }
  thrust::device_vector<float>& opacities() { return m_opacities; }
  thrust::device_vector<vec4>& rotations() { return m_rotations; }
  thrust::device_vector<vec3>& scales() { return m_scales; }

  /// @brief Per-degree SH coefficient buffers (SoA float layout on GPU).
  /// @{
  const thrust::device_vector<float>& sh0() const { return m_sh0; }
  const thrust::device_vector<float>& sh1() const { return m_sh1; }
  const thrust::device_vector<float>& sh2() const { return m_sh2; }
  const thrust::device_vector<float>& sh3() const { return m_sh3; }
  thrust::device_vector<float>& sh0() { return m_sh0; }
  thrust::device_vector<float>& sh1() { return m_sh1; }
  thrust::device_vector<float>& sh2() { return m_sh2; }
  thrust::device_vector<float>& sh3() { return m_sh3; }
  /// @}

  /// @brief Get raw device pointer to a specific SH degree buffer.
  /// @param degree SH degree (0-3).
  float* sh_degree_data(int degree);
  const float* sh_degree_data(int degree) const;

  /// @brief Number of float elements per SH degree buffer = num_coeffs * 3 * N.
  static constexpr int sh_degree_num_coeffs(int degree) {
    constexpr int coeffs[] = {1, 3, 5, 7};
    return coeffs[degree];
  }

  /// @brief Total floats in the SH degree buffer for N gaussians.
  int sh_degree_buffer_size(int degree) const {
    return sh_degree_num_coeffs(degree) * 3 * static_cast<int>(size());
  }

  std::unique_ptr<GPUGaussian3d> clone_async(BackendStream stream = nullptr);
  std::unique_ptr<GPUGaussian3d> clone();

  void memset_async(char value, BackendStream stream);
  void memset(char value);

  void remove(char* kept_flag, int num_kept);

  /// @brief Reorder gaussians according to the indices.
  //         It performs a gather: new[i] = old[indices[i]]
  /// @note the indices buffer must be on device memory.
  void reorder(uint* indices, BackendStream stream = nullptr);

  void append(int num_dup);

  float scene_scale() const { return m_scene_scale; }

  void set_scene_scale(float scale) { m_scene_scale = scale; }

  void set_sh_degree(int degree) { m_current_sh_degree = std::min(kMaxSphericalHarmonicsDegree, degree); }

  int get_sh_degree() const { return m_current_sh_degree; }

private:
  int m_current_sh_degree = 0;
  float m_scene_scale = 1.0f;

  thrust::device_vector<vec3> m_means;       ///< 3D positions
  thrust::device_vector<float> m_opacities;  ///< Opacity values (logit space)
  thrust::device_vector<vec4> m_rotations;   ///< Rotation quaternions, stored as (w, x, y, z)
  thrust::device_vector<vec3> m_scales;      ///< Scale factors (log space)

  /// SH coefficient buffers: channel-first SoA layout (RR..GG..BB per coefficient).
  /// Size of each buffer = num_coeffs_for_degree * 3 * N.
  thrust::device_vector<float> m_sh0;  ///< Degree 0: 1 coeff, size = 3*N
  thrust::device_vector<float> m_sh1;  ///< Degree 1: 3 coeffs, size = 9*N
  thrust::device_vector<float> m_sh2;  ///< Degree 2: 5 coeffs, size = 15*N
  thrust::device_vector<float> m_sh3;  ///< Degree 3: 7 coeffs, size = 21*N
};

}  // namespace tinygs
