/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   common_device.h
 *  @author Thomas Müller & Nikolaus Binder, NVIDIA
 *  @brief  Implementation of various miscellaneous CUDA kernels and
            device functions.
 */

#pragma once

#include <tinygs/common.hpp>

#include <tinygs/random/pcg32.hpp>

namespace tinygs {


__forceinline__ __device__ unsigned lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

#define IQ_DEFAULT_STATE 0x853c49e6748fea9bULL

/// Based on https://www.iquilezles.org/www/articles/sfrand/sfrand.htm
struct iqrand {
  /// @brief Initialize with default seed
  TINYGS_HOST_DEVICE iqrand() : state((uint32_t)IQ_DEFAULT_STATE) {}

  /// @brief Initialize with custom seed
  TINYGS_HOST_DEVICE iqrand(uint32_t initstate) : state(initstate) {}

  /// @brief Generate float in [0, 1)
  TINYGS_HOST_DEVICE float next_float() {
    union {
      float fres;
      unsigned int ires;
    };

    state *= 16807;
    ires = ((((unsigned int)state) >> 9) | 0x3f800000);
    return fres - 1.0f;
  }

  uint32_t state; ///< RNG state
};

using default_rng_t = pcg32;

__device__ inline float random_val(uint32_t seed, uint32_t idx) {
  default_rng_t rng{seed};
  rng.advance(idx);
  return rng.next_float();
}

__device__ inline float smoothstep(float val) {
  return val * val * (3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
  return 6 * val * (1.0f - val);
}

__device__ inline float smoothstep_2nd_derivative(float val) {
  return 6.0f - 12.0f * val;
}

__device__ inline float identity_fun(float val) { return val; }

__device__ inline float identity_derivative(float val) { return 1.0f; }

__device__ inline float identity_2nd_derivative(float val) { return 0.0f; }

__device__ inline float gaussian_cdf(const float x, const float inv_radius) {
  return normcdff(x * inv_radius);
}

__device__ inline float gaussian_cdf_approx(const float x,
                                            const float inv_radius) {
  static constexpr float MAGIC_SIGMOID_FACTOR = 1.12f / SQRT2;
  return logistic(MAGIC_SIGMOID_FACTOR * x * inv_radius);
}

__device__ inline float gaussian_cdf_approx_derivative(const float result,
                                                       const float inv_radius) {
  static constexpr float MAGIC_SIGMOID_FACTOR = 1.12f / SQRT2;
  return result * (1 - result) * MAGIC_SIGMOID_FACTOR * inv_radius;
}

__device__ inline float gaussian_pdf(const float x, const float inv_radius) {
  return inv_radius * rsqrtf(2.0f * PI()) *
         expf(-0.5f * (x * x * inv_radius * inv_radius));
}

__device__ inline float gaussian_pdf_max_1(const float x,
                                           const float inv_radius) {
  return expf(-0.5f * (x * x * inv_radius * inv_radius));
}

__device__ inline float tent(const float x, const float inv_radius) {
  return fmaxf(1.0f - fabsf(x * inv_radius), 0.0f);
}

__device__ inline float tent_cdf(const float x, const float inv_radius) {
  return fmaxf(0.0f, fminf(1.0f, x * inv_radius + 0.5f));
}

__host__ __device__ inline float quartic(const float x,
                                         const float inv_radius) {
  const float u = x * inv_radius;
  const float tmp = fmaxf(1 - u * u, 0.0f);
  return ((float)15 / 16) * tmp * tmp;
}

__host__ __device__ inline float quartic_cdf_deriv(const float x,
                                                   const float inv_radius) {
  return quartic(x, inv_radius) * inv_radius;
}

__host__ __device__ inline float quartic_cdf(const float x,
                                             const float inv_radius) {
  const float u = x * inv_radius;
  const float u2 = u * u;
  const float u4 = u2 * u2;
  return fmaxf(0.0f, fminf(1.0f, ((float)15 / 16) * u *
                                         (1 - ((float)2 / 3) * u2 +
                                          ((float)1 / 5) * u4) +
                                     0.5f));
}

__device__ __forceinline__ float saturate(float x) {
  return __saturatef(x); // Uses CUDA intrinsic for faster clamping
}

__device__ __forceinline__ float saturate_deriv(float x) {
  // Derivative of saturate: 1 inside (0,1), 0 otherwise
  return (x > 0.0f && x < 1.0f) ? 1.0f : 0.0f;
}


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ inline uint32_t expand_bits(uint32_t v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t xx = expand_bits(x);
	uint32_t yy = expand_bits(y);
	uint32_t zz = expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

__host__ __device__ inline uint32_t morton3D_invert(uint32_t x) {
	x = x               & 0x49249249;
	x = (x | (x >> 2))  & 0xc30c30c3;
	x = (x | (x >> 4))  & 0x0f00f00f;
	x = (x | (x >> 8))  & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

} // namespace tinygs