/// @file sh_pack_utils.cuh
/// @brief Utility kernels for converting between SoA (GPU storage) and AoS (rasterizer kernel)
///        SH coefficient formats. Used by rasterizer wrappers to bridge the SoA GPU layout
///        with the AoS format expected by the rasterizer kernel internals.
///
/// SoA layout (GPU storage): for degree d with C coefficients:
///   index = (k * 3 + channel) * N + gaussian_idx
///   where k = coefficient, channel = 0(R)/1(G)/2(B)
///
/// AoS layout (rasterizer kernel): float3 per coefficient per Gaussian:
///   sh_coefficient_0[gaussian_idx] = {R, G, B}  (band 0, 1 coeff)
///   sh_coefficients_rest[gaussian_idx * 15 + k] = {R, G, B}  (bands 1-3, 15 coeffs)

#pragma once
#include <cuda_runtime.h>

namespace tinygs {

// ============================================================================
// SoA → AoS packing (for forward: GPU storage → rasterizer kernels)
// ============================================================================

/// @brief Pack sh0 SoA [3*N floats] → AoS [N float3].
__global__ inline void pack_sh0_soa_to_aos(
    const float* __restrict__ sh0_soa,
    float3* __restrict__ sh0_aos,
    int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  sh0_aos[i] = make_float3(
      sh0_soa[0 * N + i],   // R
      sh0_soa[1 * N + i],   // G
      sh0_soa[2 * N + i]);  // B
}

/// @brief Pack sh1+sh2+sh3 SoA buffers → single AoS sh_coefficients_rest [N*15 float3].
///
/// The rest buffer layout has 15 float3 per Gaussian:
///   [0..2]  = band 1 (3 coefficients from sh1)
///   [3..7]  = band 2 (5 coefficients from sh2)
///   [8..14] = band 3 (7 coefficients from sh3)
__global__ inline void pack_sh_rest_soa_to_aos(
    const float* __restrict__ sh1_soa,  // 9*N floats (3 coeffs × 3 channels × N)
    const float* __restrict__ sh2_soa,  // 15*N floats (5 coeffs × 3 channels × N)
    const float* __restrict__ sh3_soa,  // 21*N floats (7 coeffs × 3 channels × N)
    float3* __restrict__ rest_aos,      // N*15 float3
    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * 15) return;

  int i = idx / 15;  // Gaussian index
  int k = idx % 15;  // Coefficient index in rest buffer

  float3 val;
  if (k < 3) {
    // Band 1: coefficients 0-2 from sh1
    int ck = k;
    val = make_float3(
        sh1_soa[(ck * 3 + 0) * N + i],
        sh1_soa[(ck * 3 + 1) * N + i],
        sh1_soa[(ck * 3 + 2) * N + i]);
  } else if (k < 8) {
    // Band 2: coefficients 3-7 from sh2
    int ck = k - 3;
    val = make_float3(
        sh2_soa[(ck * 3 + 0) * N + i],
        sh2_soa[(ck * 3 + 1) * N + i],
        sh2_soa[(ck * 3 + 2) * N + i]);
  } else {
    // Band 3: coefficients 8-14 from sh3
    int ck = k - 8;
    val = make_float3(
        sh3_soa[(ck * 3 + 0) * N + i],
        sh3_soa[(ck * 3 + 1) * N + i],
        sh3_soa[(ck * 3 + 2) * N + i]);
  }
  rest_aos[i * 15 + k] = val;
}

// ============================================================================
// AoS → SoA unpacking (for backward: rasterizer kernel gradients → GPU storage)
// ============================================================================

/// @brief Unpack + accumulate AoS grad_sh0 [N float3] → SoA grad_sh0 [3*N floats].
__global__ inline void unpack_sh0_aos_to_soa(
    const float3* __restrict__ grad_sh0_aos,
    float* __restrict__ grad_sh0_soa,
    int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float3 g = grad_sh0_aos[i];
  grad_sh0_soa[0 * N + i] += g.x;
  grad_sh0_soa[1 * N + i] += g.y;
  grad_sh0_soa[2 * N + i] += g.z;
}

/// @brief Unpack + accumulate AoS grad_sh_rest [N*15 float3] → SoA grad_sh1/sh2/sh3.
__global__ inline void unpack_sh_rest_aos_to_soa(
    const float3* __restrict__ grad_rest_aos,
    float* __restrict__ grad_sh1_soa,
    float* __restrict__ grad_sh2_soa,
    float* __restrict__ grad_sh3_soa,
    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * 15) return;

  int i = idx / 15;
  int k = idx % 15;

  float3 g = grad_rest_aos[i * 15 + k];

  if (k < 3) {
    int ck = k;
    grad_sh1_soa[(ck * 3 + 0) * N + i] += g.x;
    grad_sh1_soa[(ck * 3 + 1) * N + i] += g.y;
    grad_sh1_soa[(ck * 3 + 2) * N + i] += g.z;
  } else if (k < 8) {
    int ck = k - 3;
    grad_sh2_soa[(ck * 3 + 0) * N + i] += g.x;
    grad_sh2_soa[(ck * 3 + 1) * N + i] += g.y;
    grad_sh2_soa[(ck * 3 + 2) * N + i] += g.z;
  } else {
    int ck = k - 8;
    grad_sh3_soa[(ck * 3 + 0) * N + i] += g.x;
    grad_sh3_soa[(ck * 3 + 1) * N + i] += g.y;
    grad_sh3_soa[(ck * 3 + 2) * N + i] += g.z;
  }
}

}  // namespace tinygs
