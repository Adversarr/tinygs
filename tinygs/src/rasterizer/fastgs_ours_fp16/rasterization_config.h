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
  DEF int tile_height = 16;
  DEF int block_size_blend = tile_width * tile_height;   // 256
  DEF int block_size_blend_mask = block_size_blend - 1;  // 255
  DEF int n_sequential_threshold = 8;

  DEF int blend_bwd_n_warps = 8; // number of warps per block
  DEF int blend_bwd2_n_warps = 4; // number of warps per block

  DEF float math_pi = 3.14159265358979323846f;
} // namespace tinygs::fast_gs_fp16::config

namespace config = tinygs::fast_gs_fp16::config;

namespace tinygs::fast_gs_fp16 {

// 12B = 3bank, really good alignment for shared memory
struct alignas(4) PrimitiveInfo {
  __half2_raw conic_xy;             // 4B
  __half2_raw conic_z_raw_opacity;  // 4B
  uchar3 rgb;                   // 3B, typically in [0, 255)
};

static_assert(std::is_trivially_copyable_v<PrimitiveInfo>, "PrimitiveInfo must be trivially copyable");

__device__ __forceinline__ void fast_copy(PrimitiveInfo& dst, const PrimitiveInfo& src) {
  // dst = src;
  float3& dst_rgb = reinterpret_cast<float3&>(dst);
  const float3& src_rgb = reinterpret_cast<const float3&>(src);
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

}

#undef DEF

// 256.0 half
#define TINYGS_SCALE_TRANSMITTANCE_HALF __ushort_as_half((unsigned short)0x5C00U)

// 1/256.0 half
#define TINYGS_UNSCALE_TRANSMITTANCE_HALF __ushort_as_half((unsigned short)0x1C00U)