/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DEF inline constexpr

namespace tinygs::fast_gs_fp16::config {
#ifdef NDEBUG
  DEF bool debug = false;
#else
  DEF bool debug = true;
#endif
  // rendering constants
  DEF float dilation = 0.3f;
  DEF float min_alpha_threshold_rcp = 255.0f;
  DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
  DEF float min_alpha_threshold_deactivated = -5.537334267018537f; // log(255 - 1.0)
  DEF float max_fragment_alpha = 0.99f;                          // 0.99f in original 3dgs
  DEF float transmittance_threshold = 1e-4f;
  // block size constants
  DEF int block_size_preprocess = 128;
  DEF int block_size_preprocess_backward = 128;
  DEF int block_size_apply_depth_ordering = 256;
  DEF int block_size_create_instances = 256;
  DEF int block_size_extract_instance_ranges = 256;
  DEF int block_size_extract_bucket_counts = 256;
  DEF int tile_width = 16;
  DEF int tile_width_minus_1 = tile_width - 1;
  DEF int tile_width_log2 = 4; // log2(16) = 4
  DEF int block_size_blend = tile_width * tile_width;   // 256
  DEF int block_size_blend_mask = block_size_blend - 1;  // 255
  DEF int n_sequential_threshold = 8;

  DEF int blend_bwd_n_warps = 4; // number of warps per block
  DEF int blend_bwd2_n_warps = 4; // number of warps per block

  DEF ushort max_contributions = 0xFFFFU; // 65535

  DEF float math_pi = 3.14159265358979323846f;
} // namespace tinygs::fast_gs_fp16::config

namespace config = tinygs::fast_gs_fp16::config;

namespace tinygs::fast_gs_fp16 {

struct alignas(16) PrimitiveInfo {
  __half2_raw conic_xy;             // 4B
  __half2_raw conic_z_opacity;  // 4B
  uchar3 rgb;                   // 3B, typically in [0, 255)
};


// 256.0 half
#define TINYGS_SCALE_FULL 255.0f
// #define TINYGS_SCALE_HALF __ushort_as_half((unsigned short)0x5C00U)
#define TINYGS_SCALE_HALF __float2half_rn(TINYGS_SCALE_FULL)
#define TINYGS_SCALE_HALF2 make_half2(TINYGS_SCALE_HALF, TINYGS_SCALE_HALF)
// 1/256.0 half
#define TINYGS_UNSCALE_FULL (1.0f / TINYGS_SCALE_FULL)
// #define TINYGS_UNSCALE_HALF __ushort_as_half((unsigned short)0x1C00U)
#define TINYGS_UNSCALE_HALF __float2half_rn(TINYGS_UNSCALE_FULL)
#define TINYGS_UNSCALE_HALF2 make_half2(TINYGS_UNSCALE_HALF, TINYGS_UNSCALE_HALF)

struct alignas(8) packed_half2x2 {
  __half2 xy;
  __half2 zw;
};

struct alignas(4) PrimitiveInfoGradient {
  __half2 mean_xy;
  __half2 conic_ab;
  __half2 conic_c_color_b;
  __half2 color_rg;
  __half2 absmean_xy;
};

__device__ __forceinline__ void fast_zero(packed_half2x2& p) {
  reinterpret_cast<uint64_t&>(p) = 0ull;
}

__device__ __forceinline__
void fast_copy(packed_half2x2& dst, const packed_half2x2& src) {
  reinterpret_cast<uint64_t&>(dst) = reinterpret_cast<const uint64_t&>(src);
}

#define TINYGS_HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define TINYGS_HALF2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

using ColorTransmittance = packed_half2x2;

static_assert(std::is_trivially_copyable_v<PrimitiveInfo>, "PrimitiveInfo must be trivially copyable");

__device__ __forceinline__ void fast_copy(PrimitiveInfo& dst, const PrimitiveInfo& src) {
  // dst = src;
  uint4& dst_rgb = reinterpret_cast<uint4&>(dst);
  const uint4& src_rgb = reinterpret_cast<const uint4&>(src);
  dst_rgb = src_rgb;
}

__device__ __forceinline__ void float32uchar3(uchar3& uc, const float3& c) {
  uc.x = static_cast<unsigned char>(__saturatef(c.x) * 255.0f);
  uc.y = static_cast<unsigned char>(__saturatef(c.y) * 255.0f);
  uc.z = static_cast<unsigned char>(__saturatef(c.z) * 255.0f);
}

__device__ __forceinline__ void uchar32float3(float3& f, const uchar3& uc) {
  constexpr float inv_255 = 1.0f / 255.0f;
  f.x = static_cast<float>(uc.x) * inv_255;
  f.y = static_cast<float>(uc.y) * inv_255;
  f.z = static_cast<float>(uc.z) * inv_255;
}

__device__ __forceinline__ uint32_t half2asui32(__half2 h) {
  return reinterpret_cast<const uint32_t&>(h);
}

__device__ __forceinline__ __half2 ui32ashalf2(uint32_t u) {
  return reinterpret_cast<const __half2&>(u);
}

// Aligned store 2 packed_half2x2 (4 half2)
__device__ __forceinline__ 
void store4a(packed_half2x2* dst, __half2 x, __half2 y, __half2 z, __half2 w) {
  asm("st.global.v4.u32 [%0], {%1, %2, %3, %4};" 
      :
      : "l"(dst),
        "r"(TINYGS_HALF2_TO_CUI(x)),
        "r"(TINYGS_HALF2_TO_CUI(y)),
        "r"(TINYGS_HALF2_TO_CUI(z)),
        "r"(TINYGS_HALF2_TO_CUI(w))
      : "memory");
}

__device__ __forceinline__
void load4a(const packed_half2x2* src, __half2& x, __half2& y, __half2& z, __half2& w) {
  const uint4 u = *(reinterpret_cast<const uint4*>(src));
  x = ui32ashalf2(u.x);
  y = ui32ashalf2(u.y);
  z = ui32ashalf2(u.z);
  w = ui32ashalf2(u.w);
}

__device__ __forceinline__ uint2 tile_linear_to_xy(uint linear, ushort2 wh) {
  return make_uint2(linear % wh.x, linear / wh.x);
}

__device__ __forceinline__ uint tile_xy_to_linear(uint2 wh, uint2 xy) {
  return wh.x * xy.y + xy.x;
}

__device__ __forceinline__ __half2 fast_exp_approx(__half2 input) {
    __half2 output;
    const __half2 log2_e = __floats2half2_rn(1.4426950409f, 1.4426950409f);
    __half2 scaled_input = __hmul2(input, log2_e);
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(TINYGS_HALF2_TO_UI(output)) : "r"(TINYGS_HALF2_TO_CUI(scaled_input)));
    return output;
}

// If both x, y are positive(including inf), we can safely use the integer comparison.
__device__ __forceinline__ uint32_t hge2_positive(__half2 x, __half2 y) {
    return __vsetgeu2(TINYGS_HALF2_TO_CUI(x), TINYGS_HALF2_TO_CUI(y));
}

}

#undef DEF