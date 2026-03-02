/// @file sh_soa_utils.cuh
/// @brief Device utility functions for direct SoA access to SH coefficients.
///
/// SoA layout: buffer[(coeff_idx * 3 + channel) * N + gaussian_idx]
///   sh0: [3*N]  — band 0 DC (1 coefficient)
///   sh1: [9*N]  — band 1 (3 coefficients)
///   sh2: [15*N] — band 2 (5 coefficients)
///   sh3: [21*N] — band 3 (7 coefficients)

#pragma once
#include <cuda_runtime.h>

namespace tinygs {

/// @brief Read a float3 SH coefficient from SoA layout.
///   Returns { buffer[(k*3+0)*N+i], buffer[(k*3+1)*N+i], buffer[(k*3+2)*N+i] }
__device__ __forceinline__ float3 read_sh_soa(
    const float* __restrict__ buffer, int k, int N, int i) {
  return make_float3(
      buffer[(k * 3 + 0) * N + i],
      buffer[(k * 3 + 1) * N + i],
      buffer[(k * 3 + 2) * N + i]);
}

/// @brief Read the DC (band 0, coeff_idx=0) SH coefficient from SoA sh0 buffer.
__device__ __forceinline__ float3 read_sh0_soa(
    const float* __restrict__ sh0, int N, int i) {
  return make_float3(sh0[0 * N + i], sh0[1 * N + i], sh0[2 * N + i]);
}

/// @brief Accumulate a float3 gradient into SoA layout (non-atomic, one-thread-per-Gaussian).
__device__ __forceinline__ void accum_sh_soa(
    float* __restrict__ buffer, int k, int N, int i, const float3& val) {
  buffer[(k * 3 + 0) * N + i] += val.x;
  buffer[(k * 3 + 1) * N + i] += val.y;
  buffer[(k * 3 + 2) * N + i] += val.z;
}

/// @brief Accumulate the DC gradient into SoA sh0 buffer.
__device__ __forceinline__ void accum_sh0_soa(
    float* __restrict__ grad_sh0, int N, int i, const float3& val) {
  grad_sh0[0 * N + i] += val.x;
  grad_sh0[1 * N + i] += val.y;
  grad_sh0[2 * N + i] += val.z;
}

/// @brief Write (assign) a float3 into SoA layout (one-thread-per-Gaussian, no accumulation).
__device__ __forceinline__ void write_sh_soa(
    float* __restrict__ buffer, int k, int N, int i, const float3& val) {
  buffer[(k * 3 + 0) * N + i] = val.x;
  buffer[(k * 3 + 1) * N + i] = val.y;
  buffer[(k * 3 + 2) * N + i] = val.z;
}

/// @brief Write (assign) the DC into SoA sh0 buffer.
__device__ __forceinline__ void write_sh0_soa(
    float* __restrict__ sh0, int N, int i, const float3& val) {
  sh0[0 * N + i] = val.x;
  sh0[1 * N + i] = val.y;
  sh0[2 * N + i] = val.z;
}

}  // namespace tinygs
