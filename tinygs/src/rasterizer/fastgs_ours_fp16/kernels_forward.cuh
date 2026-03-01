/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "tinygs/core/gaussian.hpp"

#include <cuda/pipeline>
// Disables `pipeline_shared_state` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "buffer_utils.h"
#include "../../helper_math.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
#include "tinygs/common.hpp"

#include <mma.h>

namespace cg = cooperative_groups;
namespace tinygs::fast_gs_fp16::kernels::forward {

__device__ float3 convert_sh_to_color(
    const float3* sh_coefficients_0,
    const float3* sh_coefficients_rest,
    const float3& position,
    const float3& cam_position,
    const uint primitive_idx,
    const uint active_sh_bases,
    const uint total_bases_sh_rest) {
    // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    float3 result = 0.5f + 0.28209479177387814f * sh_coefficients_0[primitive_idx];
    if (active_sh_bases > 1) {
        const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * total_bases_sh_rest;
        auto [x, y, z] = normalize(position - cam_position);
        result = result + (-0.48860251190291987f * y) * coefficients_ptr[0] + (0.48860251190291987f * z) * coefficients_ptr[1] + (-0.48860251190291987f * x) * coefficients_ptr[2];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            result = result + (1.0925484305920792f * xy) * coefficients_ptr[3] + (-1.0925484305920792f * yz) * coefficients_ptr[4] + (0.94617469575755997f * zz - 0.31539156525251999f) * coefficients_ptr[5] + (-1.0925484305920792f * xz) * coefficients_ptr[6] + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * coefficients_ptr[7];
            if (active_sh_bases > 9) {
                result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * coefficients_ptr[8] + (2.8906114426405538f * xy * z) * coefficients_ptr[9] + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * coefficients_ptr[10] + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * coefficients_ptr[11] + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * coefficients_ptr[12] + (1.4453057213202769f * z * (xx - yy)) * coefficients_ptr[13] + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * coefficients_ptr[14];
            }
        }
    }
    return result;
}


// based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L131
__device__ inline bool will_primitive_contribute(
    const float2& mean,
    const float3& conic,
    const uint tile_x,
    const uint tile_y,
    const float power_threshold) {
    const float2 rect_min = make_float2(static_cast<float>(tile_x * config::tile_width), static_cast<float>(tile_y * config::tile_width));
    const float2 rect_max = make_float2(static_cast<float>((tile_x + 1) * config::tile_width - 1), static_cast<float>((tile_y + 1) * config::tile_width - 1));

    const float x_min_diff = rect_min.x - mean.x;
    const float x_left = static_cast<float>(x_min_diff > 0.0f);
    const float not_in_x_range = x_left + static_cast<float>(mean.x > rect_max.x);
    const float y_min_diff = rect_min.y - mean.y;
    const float y_above = static_cast<float>(y_min_diff > 0.0f);
    const float not_in_y_range = y_above + static_cast<float>(mean.y > rect_max.y);

    // let's hope the compiler optimizes this properly
    if (not_in_y_range + not_in_x_range == 0.0f) {
        return true;
    }
    const float2 closest_corner = make_float2(
        fast_lerp(rect_max.x, rect_min.x, x_left),
        fast_lerp(rect_max.y, rect_min.y, y_above));
    const float2 diff = mean - closest_corner;

    const float2 d = make_float2(
        copysignf(static_cast<float>(config::tile_width - 1), x_min_diff),
        copysignf(static_cast<float>(config::tile_width - 1), y_min_diff));
    const float2 t = make_float2(
        not_in_y_range * __saturatef((d.x * conic.x * diff.x + d.x * conic.y * diff.y) / (d.x * conic.x * d.x)),
        not_in_x_range * __saturatef((d.y * conic.y * diff.x + d.y * conic.z * diff.y) / (d.y * conic.z * d.y)));
    const float2 max_contribution_point = closest_corner + t * d;
    const float2 delta = mean - max_contribution_point;
    const float max_power_in_tile = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
    return max_power_in_tile <= power_threshold;
}

__device__ __forceinline__ __half2 h2copysign(const __half2& x, const __half2& y) {
    // Reinterpret __half2 as a 32-bit unsigned integer
    uint32_t ix = reinterpret_cast<const uint32_t&>(x);
    uint32_t iy = reinterpret_cast<const uint32_t&>(y);
    // Sign-bit mask for both halfs (bit 31 and bit 15)
    const uint32_t sign_mask = 0x80008000;
    // 1. Clear sign bits of x: ix & ~sign_mask
    // 2. Extract sign bits of y: iy & sign_mask
    // 3. Merge y's sign bits into x
    uint32_t result_int = (ix & ~sign_mask) | (iy & sign_mask);
    // Reinterpret result back to __half2
    return reinterpret_cast<__half2&>(result_int);
}

__device__ static __forceinline__ __half2 h2lerp(__half2 v0, __half2 v1, __half2 t) {
  return __hfma2(t, v1, __hfma2(-t, v0, v0));
}


struct alignas(8) ConicOpacity {
    __half2 xy;
    __half2 zw;
};

__device__ ConicOpacity make_conic_opacity(__half2 xy, __half2 zw) {
    return ConicOpacity{xy, zw};
}

__device__ ConicOpacity make_conic_opacity(float3 conic, float opacity) {
  return ConicOpacity{__float22half2_rn(make_float2(conic.x, conic.y)),
                      __float22half2_rn(make_float2(conic.z, opacity))};
}


__device__ __forceinline__ bool will_primitive_contribute_half(
    float2 mean, ConicOpacity conic,
    const uint tile_x, const uint tile_y,
    const float power_threshold) {
    //? Reference float version
    // auto f3_conic = make_float3(__half2float(conic.xy.x), __half2float(conic.xy.y), __half2float(conic.zw.x));
    // return will_primitive_contribute(mean, f3_conic, tile_x, tile_y, power_threshold);

    const __half2 one = make_half2(CUDART_ONE_FP16, CUDART_ONE_FP16);
    const __half2 zero = make_half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
    const __half2 tile_sizes = make_half2(
        __float2half_rd(((float) config::tile_width - 1.0f) / (float) config::tile_width),
        __float2half_rd(((float) config::tile_width - 1.0f) / (float) config::tile_width));
    const __half2 min_diff = make_half2(
        __float2half_rd((float) tile_x - mean.x * (1.0f / config::tile_width)),
        __float2half_rd((float) tile_y - mean.y * (1.0f / config::tile_width)));
    const __half2 max_diff = min_diff + tile_sizes;

    // rect_min.x - mean.x > 0, rect_min.y - mean.y > 0
    const __half2 x_left_y_above = __hgtu2(min_diff, zero);
    // rect_max.x - mean.x < 0, rect_max.y - mean.y < 0
    const __half2 x_right_y_below = __hltu2(max_diff, zero);
    const __half2 not_in_range = x_left_y_above + x_right_y_below;
    if (__hbeq2(not_in_range, zero)) {
        // both are zero => none of the four ineq is true => intile.
        return true;
    }
    const __half2 d = h2copysign(tile_sizes, min_diff);
    // we already includes mean in the xx_diff.
    const __half2 diff = -h2lerp(max_diff, min_diff, x_left_y_above);
    const __half2 diff_xx = make_half2(diff.x, diff.x);
    const __half2 diff_yy = make_half2(diff.y, diff.y);
    const __half2 xy = conic.xy;
    const __half2 yz = make_half2(conic.xy.y, conic.zw.x);
    const __half2 xz = make_half2(conic.xy.x, conic.zw.x);
    // NOTE: Here, we do not need to unscale by tile_size, the division will handle this.
    const __half2 t_raw = __h2div(d * xy * diff_xx + d * yz * diff_yy, d * d * xz);
    const __half2 t = __hmul2_sat(t_raw, __lowhigh2highlow(not_in_range));
    const float2 delta = __half22float2(diff - t * d) * config::tile_width; //! Recover to pixel unit.
    const float fx = __half2float(conic.xy.x);
    const float fy = __half2float(conic.xy.y);
    const float fz = __half2float(conic.zw.x);
    const float max_power_in_tile =
        0.5f * (fx * delta.x * delta.x + fz * delta.y * delta.y) +
        fy * delta.x * delta.y;
    return max_power_in_tile <= power_threshold;
}

// based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L177
__device__ uint compute_exact_n_touched_tiles(
    const float2& mean2d,
    const ConicOpacity& conic,
    const uint4& screen_bounds,
    const float power_threshold,
    const uint tile_count,
    const bool active) {
    const float2 mean2d_shifted = mean2d - 0.5f;

    uint n_touched_tiles = 0;
    if (active) {
        const uint screen_bounds_width = screen_bounds.y - screen_bounds.x;
        for (uint instance_idx = 0; instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            if (will_primitive_contribute_half(mean2d_shifted, conic, tile_x, tile_y, power_threshold))
                n_touched_tiles++;
        }
    }

    const uint lane_idx = cg::this_thread_block().thread_rank() % 32u;
    const uint warp_idx = cg::this_thread_block().thread_rank() / 32u;

    const int compute_cooperatively = active && tile_count > config::n_sequential_threshold;
    uint remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);
    if (remaining_threads == 0)
        return n_touched_tiles;

    const uint n_remaining_threads = __popc(remaining_threads);

    uint mask = remaining_threads;
    while (mask) {
        const uint current_lane = __ffs(mask) - 1;  // [0,31]
        mask &= (mask - 1);                          // 清掉最低位的置位

        const uint4 screen_bounds_coop = make_uint4(
            __shfl_sync(0xffffffffu, screen_bounds.x, current_lane),
            __shfl_sync(0xffffffffu, screen_bounds.y, current_lane),
            __shfl_sync(0xffffffffu, screen_bounds.z, current_lane),
            __shfl_sync(0xffffffffu, screen_bounds.w, current_lane));
        const uint screen_bounds_width_coop = screen_bounds_coop.y - screen_bounds_coop.x;
        const uint tile_count_coop = (screen_bounds_coop.w - screen_bounds_coop.z) * screen_bounds_width_coop;

        const float2 mean2d_shifted_coop = make_float2(
            __shfl_sync(0xffffffffu, mean2d_shifted.x, current_lane),
            __shfl_sync(0xffffffffu, mean2d_shifted.y, current_lane));
        // const float3 conic_coop = make_float3(
        //     __shfl_sync(0xffffffffu, conic.x, current_lane),
        //     __shfl_sync(0xffffffffu, conic.y, current_lane),
        //     __shfl_sync(0xffffffffu, conic.z, current_lane));
        ConicOpacity conic_coop;
        conic_coop.xy = __shfl_sync(0xffffffffu, conic.xy, current_lane);
        conic_coop.zw = __shfl_sync(0xffffffffu, conic.zw, current_lane);

        const float power_threshold_coop = __shfl_sync(0xffffffffu, power_threshold, current_lane);

        const uint remaining_tile_count = tile_count_coop - config::n_sequential_threshold;
        const int n_iterations = div_round_up(remaining_tile_count, 32u);
        for (int i = 0; i < n_iterations; i++) {
            const int instance_idx = i * 32 + lane_idx + config::n_sequential_threshold;
            const int active_current = instance_idx < tile_count_coop;
            const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
            const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
            const uint contributes =
                active_current && will_primitive_contribute_half(
                                      mean2d_shifted_coop, conic_coop, tile_x,
                                      tile_y, power_threshold_coop);
            const uint contributes_ballot = __ballot_sync(0xffffffffu, contributes);
            const uint n_contributes = __popc(contributes_ballot);
            if (lane_idx == current_lane) n_touched_tiles += n_contributes;
        }
    }
    return n_touched_tiles;
}



__global__ __launch_bounds__(config::block_size_preprocess) void preprocess_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ raw_scales,
    const float4* __restrict__ raw_rotations,
    const float* __restrict__ raw_opacities,
    const float3* __restrict__ sh_coefficients_0,
    const float3* __restrict__ sh_coefficients_rest,
    const float4* __restrict__ w2c,
    const float3* __restrict__ cam_position,
    tinygs::DensificationInfo* __restrict__ densification_info,
    uint* __restrict__ primitive_depth_keys,
    uint* __restrict__ primitive_indices,
    uint* __restrict__ primitive_n_touched_tiles,
    ushort4* __restrict__ primitive_screen_bounds,
    float2* __restrict__ primitive_mean2d,
    uint* __restrict__ n_visible_primitives,
    uint* __restrict__ n_instances,
    PrimitiveInfo* __restrict__ primitive_infos,
    const uint n_primitives,
    const uint grid_width,
    const uint grid_height,
    const uint active_sh_bases,
    const uint total_bases_sh_rest,
    const float w,
    const float h,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_, // near and far are macros in windowns
    const float far_) {
    auto primitive_idx = cg::this_grid().thread_rank();
    bool active = true;
    if (primitive_idx >= n_primitives) {
        active = false;
        primitive_idx = n_primitives - 1;
    }

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int warp_idx = block.thread_rank() / 32;
    using Load3 = cub::WarpLoad<float, 3, cub::WARP_LOAD_TRANSPOSE, 32>;
    constexpr int n_warps = config::block_size_preprocess / 32;
    // starting index of the current warp
    const int block_batch_idx = block.group_index().x * config::block_size_preprocess + 32 * warp_idx;
    const int batch_size = min(block_batch_idx + warp.size(), n_primitives) - block_batch_idx;
    __shared__ typename Load3::TempStorage load_means_storage[n_warps];

    if (active)
        primitive_n_touched_tiles[primitive_idx] = 0;

    // load 3d mean
    float3 mean3d = make_float3(0);
    Load3(load_means_storage[warp_idx]).Load(
      (float*) (means + block_batch_idx),
      reinterpret_cast<float(&)[3]>(mean3d), batch_size * 3, 0.0f);

    // z culling
    const float4 w2c_r1 = w2c[0];
    const float4 w2c_r2 = w2c[1];
    const float4 w2c_r3 = w2c[2];
    const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
    if (depth < near_ || depth > far_)
        active = false;

    // load opacity
    __half raw_opacity = CUDART_ZERO_FP16;
    if (active) {
      raw_opacity = __float2half_rn(raw_opacities[primitive_idx]);
    }

    const float f_opacity = tinygs::activate_opacity(__half2float(raw_opacity));
    if (f_opacity < config::min_alpha_threshold)
        active = false;
    const __half opacity = __float2half_rn(f_opacity);
    
    __shared__ typename Load3::TempStorage load_scales_temp[n_warps];
    // compute 3d covariance from raw scale and rotation
    float3 raw_scale = make_float3(0.0f);
    Load3(load_scales_temp[warp_idx]).Load(
      (float*) (raw_scales + block_batch_idx),
      reinterpret_cast<float (&)[3]>(raw_scale), batch_size * 3, 0.0f);

    const float3 variance = make_float3(
        tinygs::activate_scale(raw_scale.x) * tinygs::activate_scale(raw_scale.x),
        tinygs::activate_scale(raw_scale.y) * tinygs::activate_scale(raw_scale.y), 
        tinygs::activate_scale(raw_scale.z) * tinygs::activate_scale(raw_scale.z));
    float qr = 0.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
    if (active) {
      auto [r, x, y, z] = raw_rotations[primitive_idx];
      qr = r; qx = x; qy = y; qz = z;
    }

    const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
    const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
    if (q_norm_sq < 1e-8f)
        active = false;

    // early exit if whole warp is inactive
    if (__ballot_sync(0xffffffffu, active) == 0)
        return;

    const float qxx = 2.0f * qxx_raw / q_norm_sq, qyy = 2.0f * qyy_raw / q_norm_sq, qzz = 2.0f * qzz_raw / q_norm_sq;
    const float qxy = 2.0f * qx * qy / q_norm_sq, qxz = 2.0f * qx * qz / q_norm_sq, qyz = 2.0f * qy * qz / q_norm_sq;
    const float qrx = 2.0f * qr * qx / q_norm_sq, qry = 2.0f * qr * qy / q_norm_sq, qrz = 2.0f * qr * qz / q_norm_sq;
    const mat3x3 rotation = {
        1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
        qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
        qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
    const mat3x3 rotation_scaled = {
        rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,
        rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,
        rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z};
    const mat3x3_triu cov3d{
        rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,
        rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,
        rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,
        rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,
        rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,
        rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,
    };

    // compute 2d mean in normalized image coordinates
    const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
    const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

    // ewa splatting
    const float clip_left = (-0.15f * w - cx) / fx;
    const float clip_right = (1.15f * w - cx) / fx;
    const float clip_top = (-0.15f * h - cy) / fy;
    const float clip_bottom = (1.15f * h - cy) / fy;
    const float tx = clamp(x, clip_left, clip_right);
    const float ty = clamp(y, clip_top, clip_bottom);
    const float j11 = fx / depth;
    const float j13 = -j11 * tx;
    const float j22 = fy / depth;
    const float j23 = -j22 * ty;
    const float3 jw_r1 = make_float3(
        j11 * w2c_r1.x + j13 * w2c_r3.x,
        j11 * w2c_r1.y + j13 * w2c_r3.y,
        j11 * w2c_r1.z + j13 * w2c_r3.z);
    const float3 jw_r2 = make_float3(
        j22 * w2c_r2.x + j23 * w2c_r3.x,
        j22 * w2c_r2.y + j23 * w2c_r3.y,
        j22 * w2c_r2.z + j23 * w2c_r3.z);
    const float3 jwc_r1 = make_float3(
        jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
        jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
        jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
    const float3 jwc_r2 = make_float3(
        jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
        jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
        jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);
    float3 cov2d = make_float3(
        dot(jwc_r1, jw_r1),
        dot(jwc_r1, jw_r2),
        dot(jwc_r2, jw_r2));

    /// TrickGS: HW / 9Pi N
    // const float dilation = fmaxf(config::dilation, float(h * w) / (9.0f * config::math_pi * n_primitives));
    const float dilation = config::dilation;
    cov2d.x += dilation;
    cov2d.z += dilation;

    const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (determinant < 1e-8f)
        active = false;
    const float3 conic = make_float3(
        cov2d.z / determinant,
        -cov2d.y / determinant,
        cov2d.x / determinant);
    if (densification_info){
        float mid = 0.5f * (cov2d.x + cov2d.z);
        float lambda1 = mid + sqrt(max(0.1f, mid * mid - determinant));
        float lambda2 = mid - sqrt(max(0.1f, mid * mid - determinant));
        float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
        densification_info[primitive_idx].max_radii_screen = fmaxf(densification_info[primitive_idx].max_radii_screen, my_radius);
    }

    // 2d mean in screen space
    const float2 mean2d = make_float2(
        x * fx + cx,
        y * fy + cy);

    //! Handle half precision modifications
    PrimitiveInfo info;
    info.conic_xy = __float22half2_rn(make_float2(conic.x, conic.y));
    info.conic_z_opacity = make_half2(__float2half(conic.z), opacity);
    // compute bounds
    const float power_threshold = logf(__half2float(__ushort_as_half(info.conic_z_opacity.y)) * config::min_alpha_threshold_rcp);
    const float power_threshold_factor = sqrtf(2.0f * power_threshold);
    float extent_x = fmaxf(power_threshold_factor * sqrtf(cov2d.x) - 0.5f, 0.0f);
    float extent_y = fmaxf(power_threshold_factor * sqrtf(cov2d.z) - 0.5f, 0.0f);
    const uint4 screen_bounds = make_uint4(
        min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))),   // x_min
        min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))),   // x_max
        min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_width))))), // y_min
        min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_width)))))  // y_max
    );
    const uint n_touched_tiles_max = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
    if (n_touched_tiles_max == 0)
        active = false;

    // early exit if whole warp is inactive
    if (__ballot_sync(0xffffffffu, active) == 0)
        return;

    // ConicOpacity conic_opacity = make_conic_opacity(conic, __half2float(raw_opacity));
    ConicOpacity conic_opacity{info.conic_xy, info.conic_z_opacity};
    // compute exact number of tiles the primitive overlaps
    const uint n_touched_tiles = compute_exact_n_touched_tiles(
        mean2d, conic_opacity, screen_bounds,
        power_threshold, n_touched_tiles_max, active);

    // cooperative threads no longer needed
    if (n_touched_tiles == 0 || !active)
        return;

    // store results
#ifndef NDEBUG
    // Boundary check for primitive arrays
    assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
    primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
#ifndef NDEBUG
    assert(screen_bounds.x <= 0xffffu && screen_bounds.y <= 0xffffu &&
           screen_bounds.z <= 0xffffu && screen_bounds.w <= 0xffffu);
#endif
    primitive_screen_bounds[primitive_idx] = make_ushort4(
        static_cast<ushort>(screen_bounds.x),
        static_cast<ushort>(screen_bounds.y),
        static_cast<ushort>(screen_bounds.z),
        static_cast<ushort>(screen_bounds.w));
    primitive_mean2d[primitive_idx] = mean2d;
    auto color = convert_sh_to_color(
        sh_coefficients_0, sh_coefficients_rest,
        mean3d, cam_position[0],
        primitive_idx, active_sh_bases, total_bases_sh_rest);

    const uint offset = atomicAdd(n_visible_primitives, 1);
    const uint depth_key = __float_as_uint(depth);
    primitive_depth_keys[offset] = depth_key;
    primitive_indices[offset] = primitive_idx;
    atomicAdd(n_instances, n_touched_tiles);

    float32uchar3(info.rgb, color);
    fast_copy(primitive_infos[primitive_idx], info);
}

__device__ __forceinline__ uint32_t fns(uint32_t mask, uint32_t base, int offset) {
    uint32_t r;
    asm ("fns.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(mask), "r"(base), "r"(offset));
    return r;
}

__global__ void apply_depth_ordering_cu(
    const uint* primitive_indices_sorted,
    const uint* primitive_n_touched_tiles,
    uint* primitive_offset,
    const uint n_visible_primitives) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= n_visible_primitives)
        return;
    const uint primitive_idx = primitive_indices_sorted[idx];
    primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
}

// based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
__global__ void create_instances_cu(
    const uint* primitive_indices_sorted,
    const uint* primitive_offsets,
    const ushort4* primitive_screen_bounds,
    const float2* primitive_mean2d,
    uint* instance_keys,
    uint* instance_primitive_indices,
    const PrimitiveInfo* __restrict__ primitive_infos,
    const uint grid_width,
    const uint n_visible_primitives) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32u>(block);
    uint idx = cg::this_grid().thread_rank();

    bool active = true;
    if (idx >= n_visible_primitives) {
        active = false;
        idx = n_visible_primitives - 1;
    }

    if (__ballot_sync(0xffffffffu, active) == 0)
        return;

    const uint primitive_idx = primitive_indices_sorted[idx];

    const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
    const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
    const uint tile_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;

    __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
    __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
    __shared__ __half2 collected_conic_xy[config::block_size_create_instances];
    __shared__ __half2 collected_conic_z_raw_opacity[config::block_size_create_instances];
    collected_screen_bounds[block.thread_rank()] = screen_bounds;
    collected_mean2d_shifted[block.thread_rank()] = primitive_mean2d[primitive_idx] - 0.5f;
    {
      const PrimitiveInfo info = primitive_infos[primitive_idx];
      collected_conic_xy[block.thread_rank()] = info.conic_xy;
      collected_conic_z_raw_opacity[block.thread_rank()] = info.conic_z_opacity;
    }

    block.sync();

    uint current_write_offset = primitive_offsets[idx];

    if (active) {
        const float2 mean2d_shifted = collected_mean2d_shifted[block.thread_rank()];
        ConicOpacity conic = make_conic_opacity(
            collected_conic_xy[block.thread_rank()],
            collected_conic_z_raw_opacity[block.thread_rank()]);
        // const float3 conic = make_float3(conic_opacity);
        const float power_threshold = logf(__half2float(conic.zw.y) * config::min_alpha_threshold_rcp);

        for (uint instance_idx = 0; instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            if (will_primitive_contribute_half(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                const ushort tile_key = static_cast<ushort>(tile_y * grid_width + tile_x);
                instance_keys[current_write_offset] = tile_key;
                instance_primitive_indices[current_write_offset] = primitive_idx;
                current_write_offset++;
            }
        }
    }

    const uint lane_idx = cg::this_thread_block().thread_rank() % 32u;
    const uint warp_idx = cg::this_thread_block().thread_rank() / 32u;
    const uint lane_mask_allprev_excl = 0xffffffffu >> (32u - lane_idx);
    const int compute_cooperatively = active && tile_count > config::n_sequential_threshold;
    const uint remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);
    if (remaining_threads == 0)
        return;

    const uint n_remaining_threads = __popc(remaining_threads);
    for (int n = 0; n < n_remaining_threads && n < 32; n++) {
        int current_lane = fns(remaining_threads, 0, n + 1);
        uint primitive_idx_coop = __shfl_sync(0xffffffffu, primitive_idx, current_lane);
        uint current_write_offset_coop = __shfl_sync(0xffffffffu, current_write_offset, current_lane);

        const ushort4 screen_bounds_coop = collected_screen_bounds[warp.meta_group_rank() * 32 + current_lane];
        const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
        const uint tile_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);

        const float2 mean2d_shifted_coop = collected_mean2d_shifted[warp.meta_group_rank() * 32 + current_lane];
        ConicOpacity conic_opacity_coop = make_conic_opacity(
            (collected_conic_xy[warp.meta_group_rank() * 32 + current_lane]),
            (collected_conic_z_raw_opacity[warp.meta_group_rank() * 32 + current_lane]));

        const float power_threshold_coop =
            logf(__half2float(conic_opacity_coop.zw.y) *
                 config::min_alpha_threshold_rcp);

        const uint remaining_tile_count = tile_count_coop - config::n_sequential_threshold;
        const int n_iterations = div_round_up(remaining_tile_count, 32u);
        for (int i = 0; i < n_iterations; i++) {
            const int instance_idx = i * 32 + lane_idx + config::n_sequential_threshold;
            const int active_current = instance_idx < tile_count_coop;
            const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
            const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
            const uint write = active_current && will_primitive_contribute_half(mean2d_shifted_coop, conic_opacity_coop, tile_x, tile_y, power_threshold_coop);
            const uint write_ballot = __ballot_sync(0xffffffffu, write);
            const uint n_writes = __popc(write_ballot);
            const uint write_offset_current = __popc(write_ballot & lane_mask_allprev_excl);
            const uint write_offset = current_write_offset_coop + write_offset_current;
            if (write) {
                const uint tile_key_u32 = tile_y * grid_width + tile_x;
                instance_keys[write_offset] = tile_key_u32;
                instance_primitive_indices[write_offset] = primitive_idx_coop;
            }
            current_write_offset_coop += n_writes;
        }

        __syncwarp();
    }
}

__global__ void extract_instance_ranges_cu(
    const uint* instance_keys,
    uint2* tile_instance_ranges,
    const uint n_instances) {
    auto instance_idx = cg::this_grid().thread_rank();
    if (instance_idx >= n_instances)
        return;
    const uint instance_tile_idx = instance_keys[instance_idx];
    if (instance_idx == 0)
        tile_instance_ranges[instance_tile_idx].x = 0;
    else {
        const uint previous_instance_tile_idx = instance_keys[instance_idx - 1];
        if (instance_tile_idx != previous_instance_tile_idx) {
            tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
            tile_instance_ranges[instance_tile_idx].x = instance_idx;
        }
    }
    if (instance_idx == n_instances - 1)
        tile_instance_ranges[instance_tile_idx].y = n_instances;
}

__global__ void extract_bucket_counts(
    uint2* tile_instance_ranges,
    uint* tile_n_buckets,
    const uint n_tiles) {
    auto tile_idx = cg::this_grid().thread_rank();
    if (tile_idx >= n_tiles)
        return;
    const uint2 instance_range = tile_instance_ranges[tile_idx];
    const uint n_buckets = div_round_up(instance_range.y - instance_range.x, 32u);
#ifndef NDEBUG
    // Boundary check for tile arrays
    assert(tile_idx >= 0 && tile_idx < n_tiles);
#endif
    tile_n_buckets[tile_idx] = n_buckets;
}

// launch as 128, get 256 throughput, 2 pixel per thread.
__global__ void __launch_bounds__(config::block_size_blend / 2) blend_cu(
    const uint2* __restrict__ tile_instance_ranges,
    const uint* __restrict__ tile_bucket_offsets,
    const uint* __restrict__ instance_primitive_indices,
    const float2* __restrict__ primitive_mean2d,
    const PrimitiveInfo* __restrict__ primitive_infos,
    float16_t* __restrict__ image,
    float16_t* __restrict__ alpha_map,
    ushort* __restrict__ tile_max_n_contributions,
    ushort* __restrict__ tile_n_contributions,
    uint* __restrict__ bucket_tile_index,
    ColorTransmittance* __restrict__ bucket_color_transmittance_scaled,
    const uint width,
    const uint height,
    const uint grid_width,
    const uint n_tiles) {
    /**
     * TODO: Improve the memory throughput of this kernel. This version is 10% SLOWER than the float version.
     * 
     * 1. Increase the occupancy. Although we rely on the thread_rank to get the pixel coordinates,
     *    we can rewrite this logic to increase the block_size to hide latency.
     * 2. Replace the global memory IO to a more efficient, vectorized version.
     * 3. Replace the __half constants to their ushort16 version.
     */
    constexpr int block_size_total = config::tile_width * config::tile_width; // 256
    constexpr int block_size_launch = config::block_size_blend / 2; // 128

    // SIMD acceleration for ushort2.
    union simd32i {
        ushort2 data; // .x .y correspondingly.
        uint32_t data_u32;
        int32_t data_i32;
    };

    // --- Thread and block info ---
    auto block = cg::this_thread_block();
    const dim3 group_index = block.group_index();
    const dim3 thread_index = block.thread_index();
    const uint thread_rank = block.thread_rank();
    // --- Constants ---
    const __half2 hinv_16 = __float22half2_rn(make_float2(0.0625f, 0.0625f));
    const __half2 h_16_2 = __float22half2_rn(make_float2(16.0f, 16.0f));
    const __half h0_5 = __float2half_rn(0.5f);
    const __half2 h0_5_2 = make_half2(h0_5, h0_5);
    const __half2 h0_2 = make_half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
    const float2 anchor = make_float2(group_index.x, group_index.y);
    constexpr uint32_t one_u162 = 0x00010001u;
    const __half2 one_h2 = make_half2(CUDART_ONE_FP16, CUDART_ONE_FP16);
    const __half2 least_acceptable_transmittance_h2 = make_half2(__float2half_rd(config::transmittance_threshold * TINYGS_SCALE_FULL),
                                                                 __float2half_rd(config::transmittance_threshold * TINYGS_SCALE_FULL));

    // each thread is responsible for 2 pixel in the tile. (2x, y) and (2x+1, y)
    const uint2 intile = make_uint2(thread_index.x * 2, thread_index.y);
    const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + intile.x,
                                          group_index.y * config::tile_width + intile.y);
    const simd32i inside = simd32i(ushort2(
        static_cast<ushort>((pixel_coords.x < width && pixel_coords.y < height) ? 1u : 0u),
        static_cast<ushort>((pixel_coords.x + 1 < width && pixel_coords.y < height) ? 1u : 0u)));

        // in tiled coordinates
    __half2 offset_x, offset_y;
    {
        const __half2 offset0 = __hfma2(hinv_16,
            make_half2(__uint2half_rn(intile.x), __uint2half_rn(intile.y)),
            make_half2(__float2half(1.0f/32.0f), __float2half(1.0f/32.0f)));
        offset_x = make_half2(offset0.x, __hadd(hinv_16.x, offset0.x));
        offset_y = make_half2(offset0.y, offset0.y);
    }

    const uint width_in_tile = (width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint height_in_tile = (height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    const uint tile_idx = group_index.y * grid_width + group_index.x;

    // Early return if tile is out of bounds
    if (tile_idx >= n_tiles) {
        return;
    }

    const uint2 tile_range = tile_instance_ranges[tile_idx];
    const int n_points_total = tile_range.y - tile_range.x;

    // Write the bucket information.
    uint bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
    const int n_buckets = div_round_up(n_points_total, 32);
    for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank;
         n_buckets_remaining > 0;
         n_buckets_remaining -= block_size_launch, current_bucket_idx += block_size_launch) {
      if (current_bucket_idx < n_buckets)
        bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
    }

    // ===== shared memory =====
    // Relative to block anchor
    __shared__ __half2 collected_mean2d[block_size_launch];
    __shared__ __half2 collected_conic_xy[block_size_launch];
    __shared__ __half2 collected_conic_z_raw_opacity[block_size_launch];
    struct alignas(4) ColorSimd {
        uchar3 rgb;
        char padding_donotuse;
    };
    __shared__ ColorSimd collected_color[block_size_launch];

    // initialize local storage
    __half2 color_r = h0_2;
    __half2 color_g = h0_2;
    __half2 color_b = h0_2;
    __half2 transmittance = TINYGS_SCALE_HALF2;

    simd32i n_possible_contributions;
    n_possible_contributions.data_i32 = 0;
    simd32i n_contributions;
    n_contributions.data_i32 = 0;
    simd32i unfinished; // 1 => unfinished, 0 => finished
    unfinished.data_u32 = inside.data_u32;

    // collaborative loading and processing
    for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank;
         n_points_remaining > 0;
         n_points_remaining -= block_size_launch, current_fetch_idx += block_size_launch) {
        if (__syncthreads_count(unfinished.data_i32) == 0)
            break;
        // load gaussian parameters.
        if (current_fetch_idx < tile_range.y) {
            const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
            collected_mean2d[thread_rank] = __float22half2_rn(primitive_mean2d[primitive_idx] / 16.0f - anchor);
            const auto info = primitive_infos[primitive_idx];
            collected_conic_xy[thread_rank] = info.conic_xy;
            collected_conic_z_raw_opacity[thread_rank] = info.conic_z_opacity;
            collected_color[thread_rank] = ColorSimd{info.rgb, char(0)};
        }
        block.sync();
        const int current_batch_size = min(block_size_launch, n_points_remaining);
        int j;
        for (j = 0; /* !done */ unfinished.data_u32 && j < current_batch_size; ++j) {
            if (j % 32 == 0) {
                const uint off = tinygs::get_linear_index_tiled(intile.y, intile.x, 2);
                store4a(bucket_color_transmittance_scaled + bucket_offset * config::block_size_blend + off,
                    color_r, color_g, color_b, transmittance);
                bucket_offset++;
            }
            n_possible_contributions.data_u32 = __vadd2(n_possible_contributions.data_u32, unfinished.data_u32);

            // Convert parameters and computations to half precision (SIMD version)
            const __half2 conic_x = make_half2(collected_conic_xy[j].x, collected_conic_xy[j].x);
            const __half2 conic_y = make_half2(collected_conic_xy[j].y, collected_conic_xy[j].y);
            const __half2 conic_z = make_half2(collected_conic_z_raw_opacity[j].x, collected_conic_z_raw_opacity[j].x);
            const __half2 opacity_h = make_half2(collected_conic_z_raw_opacity[j].y, collected_conic_z_raw_opacity[j].y);
            const __half2 collected_mean2d_x = make_half2(collected_mean2d[j].x, collected_mean2d[j].x);
            const __half2 collected_mean2d_y = make_half2(collected_mean2d[j].y, collected_mean2d[j].y);
            const __half2 dx = __hsub2(collected_mean2d_x, offset_x);
            const __half2 dy = __hsub2(collected_mean2d_y, offset_y);
            const __half2 conic_x_dxx = __hmul2(dx, __hmul2(__hmul2(conic_x, dx), h_16_2));
            const __half2 conic_z_dyy = __hmul2(dy, __hmul2(__hmul2(conic_z, dy), h_16_2));
            const __half2 conic_y_dxy = __hmul2(dx, __hmul2(__hmul2(conic_y, dy), h_16_2));
            const __half2 quad = __hadd2(conic_x_dxx, conic_z_dyy);
            const __half2 sigma_over_2_h = __hmul2(__hfma2(h0_5_2, quad, conic_y_dxy), h_16_2);
            // no continue is triggered in original code.
            uint32_t enable_this_mask = __hge2_mask(sigma_over_2_h, h0_2) & __vcmpeq2(unfinished.data_u32, one_u162);

            // on my machine, it will cast to f32 and compute, no precision loss is here.
            const __half2 gaussian_h = fast_exp_approx(__hneg2(sigma_over_2_h));
            const __half2 alpha_raw_h = __hmul2(opacity_h, gaussian_h);
            const __half2 alpha_h = __hmin2(alpha_raw_h,
                make_half2(__float2half_ru(config::max_fragment_alpha),
                           __float2half_ru(config::max_fragment_alpha)));
            enable_this_mask &= __hge2_mask(alpha_h, make_half2(__float2half_rd(config::min_alpha_threshold),
                                                                __float2half_rd(config::min_alpha_threshold)));

            // next_transmittance = transmittance * (1 - alpha)
            __half2 next_transmittance_h = __hmul2(transmittance, __hsub2(one_h2, alpha_h));
            // next_transmittance > THRESHOLD => mask = 0xFFFF
            const uint32_t next_transmittance_acceptable_mask = __hge2_mask(
                next_transmittance_h, least_acceptable_transmittance_h2);
            // if next_transmittance_h < least_acceptable_transmittance_h2, then set unfinished to false
            unfinished.data_u32 &= next_transmittance_acceptable_mask;
            enable_this_mask &= next_transmittance_acceptable_mask;

            __half2 tah2 = __hmul2(__hmul2(transmittance, alpha_h), TINYGS_UNSCALE_HALF2);
            reinterpret_cast<uint32_t&>(tah2) &= enable_this_mask;

            ColorSimd rgb = collected_color[j];
            color_r = __hfma2(make_half2(__ushort2half_rn(rgb.rgb.x), __ushort2half_rn(rgb.rgb.x)),
                tah2, color_r);
            color_g = __hfma2(make_half2(__ushort2half_rn(rgb.rgb.y), __ushort2half_rn(rgb.rgb.y)),
                tah2, color_g);
            color_b = __hfma2(make_half2(__ushort2half_rn(rgb.rgb.z), __ushort2half_rn(rgb.rgb.z)),
                tah2, color_b);
            reinterpret_cast<uint32_t&>(transmittance) = 
                (~enable_this_mask & reinterpret_cast<const uint32_t&>(transmittance)) |
                ( enable_this_mask & reinterpret_cast<const uint32_t&>(next_transmittance_h));

            //? we set max_contributions to 0xFFFF (for each ushort). We increase the value by 1 everytime
            //? Therefore, no overflow will be caused.
            n_contributions.data_u32 = (n_possible_contributions.data_u32 &  enable_this_mask) |
                                       (n_contributions.data_u32          & ~enable_this_mask);
            // If n_contributions == 0xFFFF => set unfinished to false.
            unfinished.data_u32 &= __vcmpltu2(n_contributions.data_u32, 0xFFFF'FFFFu);
        }

        j = ((j + 31) / 32) * 32; // round up to next warp
        for (; j < current_batch_size; j += 32) {
            const uint off = tinygs::get_linear_index_tiled(intile.y, intile.x, 2);
            store4a(bucket_color_transmittance_scaled + bucket_offset * config::block_size_blend + off,
                color_r, color_g, color_b, transmittance);
            bucket_offset++;
        }
    }

    const int pixel_idx = width * pixel_coords.y + pixel_coords.x; // logical.
    const uint physical_pixel_idx = tinygs::get_linear_index_tiled(
        /* row */ pixel_coords.y,
        /* col */ pixel_coords.x,
        width_in_tile);
    // Write the buffers, the image is in [0, 1] range.
    color_r = __hmul2(color_r, TINYGS_UNSCALE_HALF2);
    color_g = __hmul2(color_g, TINYGS_UNSCALE_HALF2);
    color_b = __hmul2(color_b, TINYGS_UNSCALE_HALF2);
    transmittance = __hmul2(transmittance, TINYGS_UNSCALE_HALF2);
    if (inside.data_u32 != 0) {
        // Our allocation ensures the image physical width is a multiple of 8.
        *(reinterpret_cast<__half2*>(image + physical_pixel_idx)) = color_r;
        *(reinterpret_cast<__half2*>(image + physical_pixel_idx + channel_stride)) = color_g;
        *(reinterpret_cast<__half2*>(image + physical_pixel_idx + 2 * channel_stride)) = color_b;
        *(reinterpret_cast<__half2*>(alpha_map + physical_pixel_idx)) = __hsub2(one_h2, transmittance);
        tile_n_contributions[physical_pixel_idx] = n_contributions.data.x;
        tile_n_contributions[physical_pixel_idx + 1] = n_contributions.data.y;
    }

    // max reduce the number of contributions
    using BlockReduce = cub::BlockReduce<ushort, config::tile_width / 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_width>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    ushort max_xy = n_contributions.data.x > n_contributions.data.y ? n_contributions.data.x : n_contributions.data.y;
    max_xy = BlockReduce(temp_storage).Reduce(max_xy, [](ushort a, ushort b) { return a > b ? a : b; });

    if (thread_rank == 0) {
#ifndef NDEBUG
        // Boundary check for tile arrays
        assert(tile_idx >= 0 && tile_idx < n_tiles);
#endif
        // tile_max_n_contributions[tile_idx] = static_cast<ushort>(n_contributions);
        tile_max_n_contributions[tile_idx] = max_xy;
    }
}

// launch as 128, get 256 throughput, 2 pixel per thread.
__global__ void __launch_bounds__(config::block_size_blend / 2) blend_cu2(
    const uint2* __restrict__ tile_instance_ranges,
    const uint* __restrict__ tile_bucket_offsets,
    const uint* __restrict__ instance_primitive_indices,
    const float2* __restrict__ primitive_mean2d,
    const PrimitiveInfo* __restrict__ primitive_infos,
    float16_t* __restrict__ image,
    float16_t* __restrict__ alpha_map,
    ushort* __restrict__ tile_max_n_contributions,
    ushort* __restrict__ tile_n_contributions,
    uint* __restrict__ bucket_tile_index,
    ColorTransmittance* __restrict__ bucket_color_transmittance_scaled,
    const uint width,
    const uint height,
    const uint grid_width,
    const uint n_tiles) {
    constexpr int block_size_total = config::tile_width * config::tile_width; // 256
    constexpr int block_size_launch = config::block_size_blend / 2; // 128

    // SIMD acceleration for ushort2.
    union simd32i {
        ushort2 data; // .x .y correspondingly.
        uint32_t data_u32;
    };

    // --- Thread and block info ---
    auto block = cg::this_thread_block();
    const dim3 group_index = block.group_index();
    const dim3 thread_index = block.thread_index();
    const uint thread_rank = block.thread_rank();
    // --- Constants ---
    const __half2 hinv_16 = __float22half2_rn(make_float2(0.0625f, 0.0625f));
    const __half2 h_16_2 = __float22half2_rn(make_float2(16.0f, 16.0f));
    const __half h0_5 = __float2half_rn(0.5f);
    const __half2 h0_5_2 = make_half2(h0_5, h0_5);
    const __half2 h0_2 = make_half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
    const float2 anchor = make_float2(group_index.x, group_index.y);
    constexpr uint32_t one_u162 = 0x00010001u;
    const __half2 one_h2 = make_half2(CUDART_ONE_FP16, CUDART_ONE_FP16);
    const __half2 least_acceptable_transmittance_h2 = make_half2(__float2half_rd(config::transmittance_threshold * TINYGS_SCALE_FULL),
                                                                 __float2half_rd(config::transmittance_threshold * TINYGS_SCALE_FULL));

    // each thread is responsible for 2 pixel in the tile. (2x, y) and (2x+1, y)
    const uint2 intile = make_uint2(thread_index.x * 2, thread_index.y);
    const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + intile.x,
                                          group_index.y * config::tile_width + intile.y);
    const simd32i inside = simd32i(ushort2(
        static_cast<ushort>((pixel_coords.x < width && pixel_coords.y < height) ? 1u : 0u),
        static_cast<ushort>((pixel_coords.x + 1 < width && pixel_coords.y < height) ? 1u : 0u)));

        // in tiled coordinates
    __half2 offset_x, offset_y;
    {
        const __half2 offset0 = __hfma2(hinv_16,
            make_half2(__uint2half_rn(intile.x), __uint2half_rn(intile.y)),
            make_half2(__float2half(1.0f/32.0f), __float2half(1.0f/32.0f)));
        offset_x = make_half2(offset0.x, __hadd(hinv_16.x, offset0.x));
        offset_y = make_half2(offset0.y, offset0.y);
    }

    const uint width_in_tile = (width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint height_in_tile = (height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    const uint tile_idx = group_index.y * grid_width + group_index.x;

    // Early return if tile is out of bounds
    if (tile_idx >= n_tiles) {
        return;
    }

    const uint2 tile_range = tile_instance_ranges[tile_idx];
    const int n_points_total = tile_range.y - tile_range.x;

    // Write the bucket information.
    uint bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
    const int n_buckets = div_round_up(n_points_total, 32);
    for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank;
         n_buckets_remaining > 0;
         n_buckets_remaining -= block_size_launch, current_bucket_idx += block_size_launch) {
      if (current_bucket_idx < n_buckets)
        bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
    }

    // ===== shared memory =====
    // Relative to block anchor
    // __shared__ __half2 collected_mean2d[block_size_launch];
    __shared__ __half2 collected_mean2d_x[block_size_launch];
    __shared__ __half2 collected_mean2d_y[block_size_launch];
    __shared__ __half2 collected_conic_x[block_size_launch];
    __shared__ __half2 collected_conic_y[block_size_launch];
    __shared__ __half2 collected_conic_z[block_size_launch];
    __shared__ __half2 collected_opacity[block_size_launch];
    __shared__ __half2 collected_color_r[block_size_launch];
    __shared__ __half2 collected_color_g[block_size_launch];
    __shared__ __half2 collected_color_b[block_size_launch];

    // initialize local storage
    __half2 color_r = h0_2;
    __half2 color_g = h0_2;
    __half2 color_b = h0_2;
    __half2 transmittance = TINYGS_SCALE_HALF2;

    // simd32i n_possible_contributions;
    // n_possible_contributions.data_u32 = 0;
    simd32i n_contributions;
    n_contributions.data_u32 = 0;
    simd32i unfinished; // 1 => unfinished, 0 => finished
    unfinished.data_u32 = inside.data_u32;
    const uint off = tinygs::get_linear_index_tiled(intile.y, intile.x, 2);
    auto* const bucket_out = bucket_color_transmittance_scaled + off;
    auto store_bucket = [&] {
      store4a(bucket_out + bucket_offset * config::block_size_blend, color_r,
              color_g, color_b, transmittance);
      bucket_offset++;
    };

    const __half2 h_max_frag_alpha =
        make_half2(__float2half_rn(config::max_fragment_alpha),
                   __float2half_rn(config::max_fragment_alpha));
    const __half2 h_min_alpha_threshold = 
        make_half2(__float2half_rn(config::min_alpha_threshold),
                   __float2half_rn(config::min_alpha_threshold));
    const __half2 h_16_1_2 = make_half2(__float2half_rn(16.0f), __float2half_rn(1.0f));

    auto bool2mask = [](uint32_t a, int shift) -> uint32_t { return (a << shift) - a; };

    // collaborative loading and processing
    for (int n_points_remaining = n_points_total,
         current_fetch_idx = tile_range.x + thread_rank;
         n_points_remaining > 0; n_points_remaining -= block_size_launch,
                                 current_fetch_idx += block_size_launch) {
      if (__syncthreads_count(unfinished.data_u32) == 0)
        break;
      // load gaussian parameters.
      if (current_fetch_idx < tile_range.y) {
        const uint primitive_idx =
            instance_primitive_indices[current_fetch_idx];
        float2 xy = primitive_mean2d[primitive_idx] / 16.0f - anchor;
        collected_mean2d_x[thread_rank] = make_half2(__float2half_rn(xy.x), __float2half_rn(xy.x));
        collected_mean2d_y[thread_rank] = make_half2(__float2half_rn(xy.y), __float2half_rn(xy.y));
        PrimitiveInfo info;
        reinterpret_cast<uint4&>(info) = reinterpret_cast<const uint4&>(primitive_infos[primitive_idx]);
        info.conic_xy = __hmul2(info.conic_xy, h_16_2);
        info.conic_z_opacity = __hmul2(info.conic_z_opacity, h_16_1_2);
        collected_conic_x[thread_rank] = make_half2(__ushort_as_half(info.conic_xy.x), __ushort_as_half(info.conic_xy.x));
        collected_conic_y[thread_rank] = make_half2(__ushort_as_half(info.conic_xy.y), __ushort_as_half(info.conic_xy.y));
        collected_conic_z[thread_rank] = make_half2(__ushort_as_half(info.conic_z_opacity.x), __ushort_as_half(info.conic_z_opacity.x));
        collected_opacity[thread_rank] = make_half2(__ushort_as_half(info.conic_z_opacity.y), __ushort_as_half(info.conic_z_opacity.y));
        collected_color_r[thread_rank] = make_half2(__float2half_rn(info.rgb.x), __float2half_rn(info.rgb.x));
        collected_color_g[thread_rank] = make_half2(__float2half_rn(info.rgb.y), __float2half_rn(info.rgb.y));
        collected_color_b[thread_rank] = make_half2(__float2half_rn(info.rgb.z), __float2half_rn(info.rgb.z));
      }
      block.sync();
      const int current_batch_size = min(block_size_launch, n_points_remaining);
      constexpr int warp_size = 32;
      int i = 0;
      while (/* !done */ unfinished.data_u32 && i < current_batch_size) {
        store_bucket();
        const int j_end = min(i + warp_size, current_batch_size);
        for (int j = i; j < j_end && unfinished.data_u32; ++j) {
          const __half2 conic_x = collected_conic_x[j];
          const __half2 conic_y = collected_conic_y[j];
          const __half2 conic_z = collected_conic_z[j];
          const __half2 opacity_h = collected_opacity[j];
          const __half2 mean2d_x = collected_mean2d_x[j];
          const __half2 mean2d_y = collected_mean2d_y[j];
          const __half2 dx = __hsub2(mean2d_x, offset_x);
          const __half2 dy = __hsub2(mean2d_y, offset_y);
          const __half2 conic_x_dxx = __hmul2(dx, __hmul2(conic_x, dx));
          const __half2 conic_z_dyy = __hmul2(dy, __hmul2(conic_z, dy));
          const __half2 conic_y_dxy = __hmul2(dx, __hmul2(conic_y, dy));
          const __half2 quad = __hadd2(conic_x_dxx, conic_z_dyy);
          const __half2 sigma_over_2_h =
              __hmul2(__hfma2(h0_5_2, quad, conic_y_dxy), h_16_2);
          // no continue is triggered in original code.
          // on my machine, it will cast to f32 and compute, no precision loss
          // is here.
          const __half2 gaussian_h = fast_exp_approx(__hneg2(sigma_over_2_h));
          const __half2 alpha_raw_h = __hmul2(opacity_h, gaussian_h);
          const __half2 alpha_h = __hmin2(alpha_raw_h, h_max_frag_alpha);

          // next_transmittance = transmittance * (1 - alpha) = transmittance - transmittance * alpha
          // __half2 next_transmittance_h = __hmul2(transmittance, __hsub2(one_h2, alpha_h));
          const __half2 next_transmittance_h = __hfma2(__hneg2(alpha_h), transmittance, transmittance);

          // next_transmittance > THRESHOLD => mask = 0xFFFF
          const uint32_t next_t_acceptable_01 = hge2_positive(next_transmittance_h, least_acceptable_transmittance_h2);
          // convert it to mask.
          auto enable_this_mask = bool2mask(unfinished.data_u32 & next_t_acceptable_01, 16);

          __half2 tah2 = __hmul2(__hmul2(transmittance, alpha_h), TINYGS_UNSCALE_HALF2);
          reinterpret_cast<uint32_t &>(tah2) &= enable_this_mask;
#define bitselect(a, b, mask) ((a) ^ ((mask) & ((b) ^ (a))))

          color_r = __hfma2(collected_color_r[j], tah2, color_r);
          color_g = __hfma2(collected_color_g[j], tah2, color_g);
          color_b = __hfma2(collected_color_b[j], tah2, color_b);
          TINYGS_HALF2_TO_UI(transmittance) = bitselect(
            TINYGS_HALF2_TO_CUI(transmittance),
            TINYGS_HALF2_TO_CUI(next_transmittance_h),
            enable_this_mask);

          //? we set max_contributions to 0xFFFF (for each ushort). We increase
          //the value by 1 everytime ? Therefore, no overflow will be caused.
          n_contributions.data_u32 = __vaddus2(n_contributions.data_u32, unfinished.data_u32);
          // If n_contributions == 0xFFFF => set unfinished to false.
          unfinished.data_u32 &= next_t_acceptable_01;
        }
        unfinished.data_u32 &= __vsetltu2(n_contributions.data_u32, 0xFFFF'FFFFu);
        i += warp_size;
      }

      for (; i < current_batch_size; i += 32) {
        store_bucket();
      }
    }

    const int pixel_idx = width * pixel_coords.y + pixel_coords.x; // logical.
    const uint physical_pixel_idx = tinygs::get_linear_index_tiled(
        /* row */ pixel_coords.y,
        /* col */ pixel_coords.x,
        width_in_tile);
    // Write the buffers, the image is in [0, 1] range.
    color_r = __hmul2(color_r, TINYGS_UNSCALE_HALF2);
    color_g = __hmul2(color_g, TINYGS_UNSCALE_HALF2);
    color_b = __hmul2(color_b, TINYGS_UNSCALE_HALF2);
    transmittance = __hmul2(transmittance, TINYGS_UNSCALE_HALF2);
    if (inside.data_u32 != 0) {
        // Our allocation ensures the image physical width is a multiple of 8.
        *(reinterpret_cast<__half2*>(image + physical_pixel_idx)) = color_r;
        *(reinterpret_cast<__half2*>(image + physical_pixel_idx + channel_stride)) = color_g;
        *(reinterpret_cast<__half2*>(image + physical_pixel_idx + 2 * channel_stride)) = color_b;
        *(reinterpret_cast<__half2*>(alpha_map + physical_pixel_idx)) = __hsub2(one_h2, transmittance);
        *(reinterpret_cast<uint32_t*>(tile_n_contributions+physical_pixel_idx)) = n_contributions.data_u32;
    }

    // max reduce the number of contributions
    using BlockReduce = cub::BlockReduce<ushort, config::tile_width / 2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_width>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    ushort max_xy = n_contributions.data.x > n_contributions.data.y ? n_contributions.data.x : n_contributions.data.y;
    max_xy = BlockReduce(temp_storage).Reduce(max_xy, [](ushort a, ushort b) { return a > b ? a : b; });

    if (thread_rank == 0) {
#ifndef NDEBUG
        // Boundary check for tile arrays
        assert(tile_idx >= 0 && tile_idx < n_tiles);
#endif
        // tile_max_n_contributions[tile_idx] = static_cast<ushort>(n_contributions);
        tile_max_n_contributions[tile_idx] = max_xy;
        // typically, max_xy < 2048
        // if we have 1-alpha = 0.99 => 0.99 ** 1000 = 0.00004317, which is small enough
        // to be ignored.
        // but, does we have so much transparent gaussians?
    }
}

} // namespace tinygs::fast_gs_fp16::kernels::forward
