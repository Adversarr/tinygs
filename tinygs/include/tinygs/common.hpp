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

/** @file   common.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of
 * this framework.
 */

#pragma once

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <nlohmann/json.hpp>


#include <cuda_fp16.h>
#include <cuda_bf16.h>

////////////////////////////// CUDA Macros //////////////////////////////

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

#if defined(__CUDA_ARCH__)
#define TINYGS_PRAGMA_UNROLL _Pragma("unroll")
#else
#define TINYGS_PRAGMA_UNROLL
#endif

#ifdef __CUDACC__
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress = unsigned_compare_with_zero
#else
#pragma diag_suppress = unsigned_compare_with_zero
#endif
#endif

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
#define TINYGS_HOST_DEVICE __host__ __device__
#define TINYGS_DEVICE __device__
#define TINYGS_HOST __host__
#else
#define TINYGS_HOST_DEVICE
#define TINYGS_DEVICE
#define TINYGS_HOST
#endif

#ifndef TINYGS_MIN_GPU_ARCH
#warning TINYGS_MIN_GPU_ARCH was not defined. Using default value 75.
#define TINYGS_MIN_GPU_ARCH 75
#endif

#include <tinygs/cuda/vec.hpp>

// #if defined(__CUDA_ARCH__)
// static_assert(
//     __CUDA_ARCH__ >= TINYGS_MIN_GPU_ARCH * 10,
//     "MIN_GPU_ARCH=" STR(TINYGS_MIN_GPU_ARCH) "0 must bound __CUDA_ARCH__=" STR(
//         __CUDA_ARCH__) " from below, but doesn't.");
// #endif

namespace tinygs {
using json = nlohmann::json;

#define TINYGS_HALF_PRECISION                                                  \
  (!(TINYGS_MIN_GPU_ARCH == 61 || TINYGS_MIN_GPU_ARCH <= 52))

// TinyGS has the following behavior depending on GPU arch.
// Refer to the first row of the table at the following URL for information
// about when to pick fp16 versus fp32 precision for maximum performance.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions
//
//  GPU Arch | FullyFusedMLP supported | CUTLASS SmArch supported | Precision
// ----------|-------------------------|--------------------------|--------------------------
//     80-90 |                     yes |                       80 | __half
//        75 |                     yes |                       75 | __half 70 |
//        no |                       70 |                    __half
// 53-60, 62 |                      no |                       70 |  __half (no
// tensor cores)
//  <=52, 61 |                      no |                       70 |   float (no
//  tensor cores)

#if defined(__CUDACC__)
/// Optional: use float precision for debugging
#endif

/// The frame_id, camera_id, ... Always ui64.
using uuid_t = uint64_t;

using f32 = float;
using float16_t = nv_half;
using bf16 = nv_bfloat16;

using f162 = nv_half2;
using bf162 = nv_bfloat162;

////////////////////////////// Utility Functions //////////////////////////////

inline constexpr TINYGS_HOST_DEVICE float PI() {
  return 3.14159265358979323846f;
}

template <typename T> TINYGS_HOST_DEVICE T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <typename T> TINYGS_HOST_DEVICE T next_multiple(T val, T divisor) {
  return div_round_up(val, divisor) * divisor;
}

template <typename T> TINYGS_HOST_DEVICE T previous_multiple(T val, T divisor) {
  return (val / divisor) * divisor;
}

constexpr uint32_t BATCH_SIZE_GRANULARITY = 256;
constexpr uint32_t N_THREADS_LINEAR = 128;
constexpr uint32_t WARP_SIZE = 32;

// Lower-case constants kept for backward compatibility with user code.
constexpr uint32_t batch_size_granularity = BATCH_SIZE_GRANULARITY;
constexpr uint32_t n_threads_linear = N_THREADS_LINEAR;

template <typename T>
constexpr TINYGS_HOST_DEVICE uint32_t
n_blocks_linear(T n_elements, uint32_t n_threads = N_THREADS_LINEAR) {
  return (uint32_t)div_round_up(n_elements, (T)n_threads);
}

template <typename T> struct PitchedPtr {
  TINYGS_HOST_DEVICE PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
  TINYGS_HOST_DEVICE PitchedPtr(T *ptr, size_t stride_in_elements,
                                size_t offset = 0,
                                size_t extra_stride_bytes = 0)
      : ptr{ptr + offset},
        stride_in_bytes{stride_in_elements * sizeof(T) + extra_stride_bytes} {}

  template <typename U>
  TINYGS_HOST_DEVICE explicit PitchedPtr(PitchedPtr<U> other)
      : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

  TINYGS_HOST_DEVICE T *operator()(uint32_t y) const {
    return (T *)((const char *)ptr + y * stride_in_bytes);
  }

  TINYGS_HOST_DEVICE void operator+=(uint32_t y) {
    ptr = (T *)((const char *)ptr + y * stride_in_bytes);
  }

  TINYGS_HOST_DEVICE void operator-=(uint32_t y) {
    ptr = (T *)((const char *)ptr - y * stride_in_bytes);
  }

  TINYGS_HOST_DEVICE explicit operator bool() const { return ptr; }

  T *ptr;
  size_t stride_in_bytes;
};

template <typename T> struct Interval {
  // Inclusive start, exclusive end
  T start, end;

  TINYGS_HOST_DEVICE bool operator<(const Interval &other) const {
    // This operator is used to sort non-overlapping intervals. Since intervals
    // may be empty, the second half of the following expression is required to
    // resolve ambiguity when `end` of adjacent empty intervals is equal.
    return end < other.end || (end == other.end && start < other.start);
  }

  TINYGS_HOST_DEVICE bool overlaps(const Interval &other) const {
    return !intersect(other).empty();
  }

  TINYGS_HOST_DEVICE Interval intersect(const Interval &other) const {
    return {std::max(start, other.start), std::min(end, other.end)};
  }

  TINYGS_HOST_DEVICE bool valid() const { return end >= start; }

  TINYGS_HOST_DEVICE bool empty() const { return end <= start; }

  TINYGS_HOST_DEVICE T size() const { return end - start; }
};

// Helpful data structure to represent ray-object intersections
template <typename T> struct PayloadAndIdx {
  T t;
  int64_t idx;

  // Sort in descending order
  TINYGS_HOST_DEVICE bool operator<(const PayloadAndIdx<T> &other) {
    return t < other.t;
  }
};

constexpr int kMaxSphericalHarmonicsDegree = 3;
constexpr int kMaxSphericalHarmonicsCoefficients =
    (kMaxSphericalHarmonicsDegree + 1) * (kMaxSphericalHarmonicsDegree + 1);


static constexpr float SQRT2 = 1.41421356237309504880f;

TINYGS_HOST_DEVICE inline float logistic(const float x) {
  return 1.0f / (1.0f + expf(-x));
}

TINYGS_HOST_DEVICE inline float logit(const float x) {
  return -logf(1.0f / (fminf(fmaxf(x, 1e-9f), 1.0f - 1e-9f)) - 1.0f);
}

// =============================================================================
// Image Storage Convention (CRITICAL)
// =============================================================================
// All images in this codebase (rendered output, ground truth, loss buffers)
// use 8x8 TILED storage layout (AoSoA), NOT row-major. This includes:
//   - Rasterizer output (fwd_output.image)
//   - Dataloader output (GPUBatchOutput.image)
//   - Loss function inputs/outputs
//   - FastGS metric computation buffers
//
// Use get_linear_index_tiled(row, col, tiled_width) from common.hpp to compute
// linear indices. The tiled_width = padded_width / 8.
//
// EXCEPTION: metric_map and metric_counts in RasterizeContext are FLAT arrays
// (not tiled) since they are per-pixel/per-Gaussian flags, not images.
// =============================================================================

// AoSoA for images.
constexpr uint32_t kImageTile      = 8;
constexpr uint32_t kImageTileLog2  = 3;  // log2(8) = 3
constexpr uint32_t kImageTileMask  = kImageTile - 1; // 0b111

TINYGS_HOST_DEVICE inline uint32_t get_tile_x(uint32_t j) {
  return j >> kImageTileLog2;
}

TINYGS_HOST_DEVICE inline uint32_t get_tile_y(uint32_t i) {
  return i >> kImageTileLog2;
}

// Get pixel position within a tile
TINYGS_HOST_DEVICE inline uint32_t get_intra_x(uint32_t j) {
  return j & kImageTileMask;
}

TINYGS_HOST_DEVICE inline uint32_t get_intra_y(uint32_t i) {
  return i & kImageTileMask;
}

// Calculate linear index of tile
TINYGS_HOST_DEVICE inline uint32_t get_tile_index_tiled(uint32_t i, uint32_t j, uint32_t tiled_width) {
  return get_tile_y(i) * tiled_width + get_tile_x(j);
}

TINYGS_HOST_DEVICE inline uint32_t get_tile_index(uint32_t i, uint32_t j, uint32_t width) {
  return get_tile_index_tiled(i, j, width >> kImageTileLog2);
}


// Calculate linear offset within a tile
TINYGS_HOST_DEVICE inline uint32_t get_offset_in_tile(uint32_t i, uint32_t j) {
  return (get_intra_y(i) << kImageTileLog2) + get_intra_x(j);
}

// Final: Get linear index from (i,j) coordinates
TINYGS_HOST_DEVICE inline uint32_t get_linear_index_tiled(uint32_t i, uint32_t j, uint32_t tiled_width) {
  const uint32_t tile_idx = get_tile_index_tiled(i, j, tiled_width);
  const uint32_t offset_in_tile = get_offset_in_tile(i, j);
  return (tile_idx << (2 * kImageTileLog2)) + offset_in_tile;
}

TINYGS_HOST_DEVICE inline uint32_t get_linear_index(uint32_t i, uint32_t j, uint32_t width) {
  const uint32_t tile_idx = get_tile_index(i, j, width);
  const uint32_t offset_in_tile = get_offset_in_tile(i, j);
  return (tile_idx << (2 * kImageTileLog2)) + offset_in_tile;
}

} // namespace tinygs