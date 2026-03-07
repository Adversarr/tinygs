#pragma once

#include "tinygs/platform/buffer_view.hpp"
#include "tinygs/core/gaussian.hpp"
#include "tinygs/platform/backend_types.hpp"

#include <cstddef>
#include <memory>

namespace tinygs {

class BackendRuntime;
class BackendQueue;

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
  explicit GPUGaussian3d(BackendRuntime& runtime);

  ~GPUGaussian3d();

  BackendRuntime& runtime() const;

  void copy_from_host_async(const Gaussian3d &gaussians, BackendQueue* queue);
  void copy_from_host(const Gaussian3d &gaussians, BackendQueue* queue);

  void copy_to_host_async(Gaussian3d& gaussians, BackendQueue* queue);
  void copy_to_host(Gaussian3d& gaussians, BackendQueue* queue);

  size_t size() const;

  DeviceSpan<const vec3> means() const;
  DeviceSpan<const float> opacities() const;
  DeviceSpan<const vec4> rotations() const;
  DeviceSpan<const vec3> scales() const;

  DeviceSpan<vec3> means();
  DeviceSpan<float> opacities();
  DeviceSpan<vec4> rotations();
  DeviceSpan<vec3> scales();

  /// @brief Per-degree SH coefficient buffers (SoA float layout on GPU).
  /// @{
  DeviceSpan<const float> sh0() const;
  DeviceSpan<const float> sh1() const;
  DeviceSpan<const float> sh2() const;
  DeviceSpan<const float> sh3() const;
  DeviceSpan<float> sh0();
  DeviceSpan<float> sh1();
  DeviceSpan<float> sh2();
  DeviceSpan<float> sh3();
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

  std::unique_ptr<GPUGaussian3d> clone_async(const BackendQueue* queue = nullptr);
  std::unique_ptr<GPUGaussian3d> clone();

  void memset_async(char value, const BackendQueue* queue);
  void memset(char value);

  void remove(char* kept_flag, int num_kept, const BackendQueue* queue = nullptr);

  /// @brief Reorder gaussians according to the indices.
  //         It performs a gather: new[i] = old[indices[i]]
  /// @note the indices buffer must be on device memory.
  void reorder(uint* indices, const BackendQueue* queue = nullptr);

  std::shared_ptr<BackendBuffer> compute_morton_order_indices(const BackendQueue* queue = nullptr);

  void append(int num_dup, BackendQueue* queue);

  float scene_scale() const { return m_scene_scale; }

  void set_scene_scale(float scale) { m_scene_scale = scale; }

  void set_sh_degree(int degree) { m_current_sh_degree = std::min(kMaxSphericalHarmonicsDegree, degree); }

  int get_sh_degree() const { return m_current_sh_degree; }

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  int m_current_sh_degree = 0;
  float m_scene_scale = 1.0f;
};

std::shared_ptr<BackendBuffer> reorder_densification_info(
    const std::shared_ptr<BackendBuffer>& info,
    const uint* indices,
    size_t n,
    BackendRuntime& runtime,
  const BackendQueue* queue);

}  // namespace tinygs
