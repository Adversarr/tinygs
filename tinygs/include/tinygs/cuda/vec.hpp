/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   vec.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  CUDA-specific vec extensions built on top of backend-neutral vec math.
 */

#pragma once

#include <cuda_fp16.h>

#include <tinygs/common.hpp>

namespace tinygs {

#ifdef __CUDACC__
inline TINYGS_DEVICE __half fma(__half a, __half b, __half c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  return __hfma(a, b, c);
#else
  return __float2half(fmaf(__half2float(a), __half2float(b), __half2float(c)));
#endif
}
#endif

#if defined(__CUDACC__)
inline TINYGS_DEVICE void atomic_add_gmem_float(float* addr, float in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  int in_int = *((int*)&in);
  asm("red.relaxed.gpu.global.add.f32 [%0], %1;" : : "l"(addr), "r"(in_int));
#else
  atomicAdd(addr, in);
#endif
}

template <typename T, int N>
TINYGS_DEVICE void atomic_add(T* dst, const glm::vec<N, T>& a) {
  TINYGS_PRAGMA_UNROLL
  for (uint32_t i = 0; i < N; ++i) {
    atomicAdd(dst + i, a[i]);
  }
}

template <int N>
TINYGS_DEVICE void atomic_add_gmem(float* dst, const glm::vec<N, float>& a) {
  TINYGS_PRAGMA_UNROLL
  for (uint32_t i = 0; i < N; ++i) {
    atomic_add_gmem_float(dst + i, a[i]);
  }
}
#endif

} // namespace tinygs
