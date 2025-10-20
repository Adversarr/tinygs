/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "tinygs/core/gaussian.hpp"

#include <cuda/pipeline>
// Disables `pipeline_shared_state` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "buffer_utils.h"
#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
#include "tinygs/common.hpp"

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
    const float2 rect_min = make_float2(static_cast<float>(tile_x * config::tile_width), static_cast<float>(tile_y * config::tile_height));
    const float2 rect_max = make_float2(static_cast<float>((tile_x + 1) * config::tile_width - 1), static_cast<float>((tile_y + 1) * config::tile_height - 1));

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
        copysignf(static_cast<float>(config::tile_height - 1), y_min_diff));
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
        __float2half_rd(((float) config::tile_height - 1.0f) / (float) config::tile_height));
    const __half2 min_diff = make_half2(
        __float2half_rd((float) tile_x - mean.x * (1.0f / config::tile_width)),
        __float2half_rd((float) tile_y - mean.y * (1.0f / config::tile_height)));
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



__global__ void preprocess_cu(
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
    float3* __restrict__ primitive_color,
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

    constexpr int stages_count = 3; // rot, opa, scale
    /* == common settings ==  */
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Create a synchronization object (C++20 barrier)
    __shared__ float shm_opacities[config::block_size_preprocess];
    __shared__ float4 shm_raw_rotations[config::block_size_preprocess];
    __shared__ float3 shm_raw_scales[config::block_size_preprocess];

    const int block_batch_idx = block.group_index().x * config::block_size_preprocess;
    const int block_max_idx = min(block_batch_idx + block.size(), n_primitives);
    pipeline.producer_acquire();
    cuda::memcpy_async(block, shm_opacities, raw_opacities + block_batch_idx, sizeof(float) * (block_max_idx - block_batch_idx), pipeline);
    pipeline.producer_commit();

    pipeline.producer_acquire();
    cuda::memcpy_async(block, shm_raw_scales, raw_scales + block_batch_idx, sizeof(float3) * (block_max_idx - block_batch_idx), pipeline);
    pipeline.producer_commit();

    pipeline.producer_acquire();
    cuda::memcpy_async(block, shm_raw_rotations, raw_rotations + block_batch_idx, sizeof(float4) * (block_max_idx - block_batch_idx), pipeline);
    pipeline.producer_commit();

    if (active)
        primitive_n_touched_tiles[primitive_idx] = 0;

    // load 3d mean
    const float3 mean3d = means[primitive_idx];

    // z culling
    const float4 w2c_r1 = w2c[0];
    const float4 w2c_r2 = w2c[1];
    const float4 w2c_r3 = w2c[2];
    const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
    if (depth < near_ || depth > far_)
        active = false;

    // load opacity
    pipeline.consumer_wait();
    const __half raw_opacity = __float2half_rn(shm_opacities[block.thread_rank()]);
    pipeline.consumer_release();

    const float opacity = tinygs::activate_opacity(__half2float(raw_opacity));
    if (opacity < config::min_alpha_threshold)
        active = false;

    // compute 3d covariance from raw scale and rotation
    pipeline.consumer_wait();
    const float3 raw_scale = shm_raw_scales[block.thread_rank()];
    // const float3 raw_scale = raw_scales[primitive_idx];
    pipeline.consumer_release();
    const float3 variance = make_float3(
        tinygs::activate_scale(raw_scale.x) * tinygs::activate_scale(raw_scale.x),
        tinygs::activate_scale(raw_scale.y) * tinygs::activate_scale(raw_scale.y), 
        tinygs::activate_scale(raw_scale.z) * tinygs::activate_scale(raw_scale.z));
    pipeline.consumer_wait();
    auto [qr, qx, qy, qz] = shm_raw_rotations[block.thread_rank()];
    pipeline.consumer_release();

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

    // compute bounds
    const float power_threshold = logf(opacity * config::min_alpha_threshold_rcp);
    const float power_threshold_factor = sqrtf(2.0f * power_threshold);
    float extent_x = fmaxf(power_threshold_factor * sqrtf(cov2d.x) - 0.5f, 0.0f);
    float extent_y = fmaxf(power_threshold_factor * sqrtf(cov2d.z) - 0.5f, 0.0f);
    const uint4 screen_bounds = make_uint4(
        min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))),   // x_min
        min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))),   // x_max
        min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
        min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_height)))))  // y_max
    );
    const uint n_touched_tiles_max = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
    if (n_touched_tiles_max == 0)
        active = false;

    // early exit if whole warp is inactive
    if (__ballot_sync(0xffffffffu, active) == 0)
        return;

    ConicOpacity conic_opacity = make_conic_opacity(conic, __half2float(raw_opacity));
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
    primitive_color[primitive_idx] = convert_sh_to_color(
        sh_coefficients_0, sh_coefficients_rest,
        mean3d, cam_position[0],
        primitive_idx, active_sh_bases, total_bases_sh_rest);

    const uint offset = atomicAdd(n_visible_primitives, 1);
    const uint depth_key = __float_as_uint(depth);
    primitive_depth_keys[offset] = depth_key;
    primitive_indices[offset] = primitive_idx;
    atomicAdd(n_instances, n_touched_tiles);

    //! Handle half precision modifications
    PrimitiveInfo info;
    info.conic_xy = __float22half2_rn(make_float2(conic.x, conic.y));
    info.conic_z_raw_opacity = make_half2(__float2half(conic.z), raw_opacity);
    float32uchar3(info.rgb, primitive_color[primitive_idx]);
    fast_copy(primitive_infos[primitive_idx], info);
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
    ushort* instance_keys,
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
      collected_conic_z_raw_opacity[block.thread_rank()] = info.conic_z_raw_opacity;
    }

    block.sync();

    uint current_write_offset = primitive_offsets[idx];

    if (active) {
        const float2 mean2d_shifted = collected_mean2d_shifted[block.thread_rank()];
        ConicOpacity conic = make_conic_opacity(
            collected_conic_xy[block.thread_rank()],
            collected_conic_z_raw_opacity[block.thread_rank()]);
        // const float3 conic = make_float3(conic_opacity);
        const float power_threshold = logf(activate_opacity(__half2float(conic.zw.y)) * config::min_alpha_threshold_rcp);

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
        int current_lane = __fns(remaining_threads, 0, n + 1);
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
            logf(activate_opacity(__half2float(conic_opacity_coop.zw.y)) *
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
#ifndef NDEBUG
                assert(tile_key_u32 <= 0xffffu);
#endif
                const ushort tile_key = static_cast<ushort>(tile_key_u32);
                instance_keys[write_offset] = tile_key;
                instance_primitive_indices[write_offset] = primitive_idx_coop;
            }
            current_write_offset_coop += n_writes;
        }

        __syncwarp();
    }
}

__global__ void extract_instance_ranges_cu(
    const ushort* instance_keys,
    uint2* tile_instance_ranges,
    const uint n_instances) {
    auto instance_idx = cg::this_grid().thread_rank();
    if (instance_idx >= n_instances)
        return;
    const ushort instance_tile_idx = instance_keys[instance_idx];
    if (instance_idx == 0)
        tile_instance_ranges[instance_tile_idx].x = 0;
    else {
        const ushort previous_instance_tile_idx = instance_keys[instance_idx - 1];
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

__global__ void __launch_bounds__(config::block_size_blend) blend_cu(
    const uint2* tile_instance_ranges,
    const uint* tile_bucket_offsets,
    const uint* instance_primitive_indices,
    const float2* primitive_mean2d,
    const PrimitiveInfo* primitive_infos,
    float16_t* image,
    float16_t* alpha_map,
    ushort* tile_max_n_contributions,
    ushort* tile_n_contributions,
    uint* bucket_tile_index,
    ColorTransmittance* bucket_color_transmittance_scaled,
    const uint width,
    const uint height,
    const uint grid_width,
    const uint n_tiles) {
    auto block = cg::this_thread_block();
    const dim3 group_index = block.group_index();
    const dim3 thread_index = block.thread_index();
    const uint thread_rank = block.thread_rank();

    // each thread is responsible for a pixel in the tile.
    const uint2 intile = make_uint2(thread_index.x, thread_index.y);
    const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width  + intile.x,
                                          group_index.y * config::tile_height + intile.y);
    const bool inside = pixel_coords.x < width && pixel_coords.y < height;
    const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;

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

    uint bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
    const int n_buckets = div_round_up(n_points_total, 32); // re-computing is faster than reading from tile_n_buckets
    for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank;
         n_buckets_remaining > 0;
         n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
      if (current_bucket_idx < n_buckets)
        bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
    }

    // ===== shared memory =====
    __shared__ float2 collected_mean2d[config::block_size_blend];
    // bank conflict free storage
    __shared__ __half2 collected_conic_xy[config::block_size_blend];
    __shared__ __half2 collected_conic_z_raw_opacity[config::block_size_blend];
    struct alignas(4) ColorSimd {
        uchar3 rgb;
        char padding_donotuse;
    };
    __shared__ ColorSimd collected_color[config::block_size_blend];

    // initialize local storage
    // float3 color_pixel = make_float3(0.0f);
    // float transmittance = 1.0f;
    packed_half2x2 color_transmittance_scaled;
    fast_zero(color_transmittance_scaled);
    color_transmittance_scaled.zw.y = TINYGS_SCALE_HALF;
    uint n_possible_contributions = 0;
    uint n_contributions = 0;
    bool done = !inside;

    // collaborative loading and processing
    for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank;
         n_points_remaining > 0;
         n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
        if (__syncthreads_count(done) == config::block_size_blend)
            break;
        // load gaussian parameters.
        if (current_fetch_idx < tile_range.y) {
            const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
            collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
            const auto& info = primitive_infos[primitive_idx];
            collected_conic_xy[thread_rank] = info.conic_xy;
            collected_conic_z_raw_opacity[thread_rank] = info.conic_z_raw_opacity;
            collected_color[thread_rank] = ColorSimd{info.rgb, char(0)};
        }
        block.sync();
        const int current_batch_size = min(config::block_size_blend, n_points_remaining);
        int j;
        for (j = 0; !done && j < current_batch_size; ++j) {
            if (j % 32 == 0) {
                const uint off = tinygs::get_linear_index_tiled(intile.y, intile.x, 2);
                // const float4 current_color_transmittance = make_float4(color_pixel, transmittance);
                // ColorTransmittance ct{
                //     __float22half2_rn(make_float2(current_color_transmittance.x, current_color_transmittance.y)),
                //     __float22half2_rn(make_float2(current_color_transmittance.z, current_color_transmittance.w))
                // };
                bucket_color_transmittance_scaled[bucket_offset * config::block_size_blend + off] = color_transmittance_scaled;
                bucket_offset++;
            }
            n_possible_contributions++;
            // Convert parameters and computations to half precision
            const __half conic_x = collected_conic_xy[j].x;
            const __half conic_y = collected_conic_xy[j].y;
            const __half conic_z = collected_conic_z_raw_opacity[j].x;
            const __half opacity_h = __float2half_rn(activate_opacity(__half2float(collected_conic_z_raw_opacity[j].y)));

            const __half2 delta_h2 = __hsub2(__float22half2_rn(collected_mean2d[j]), __float22half2_rn(pixel));
            const __half dx = delta_h2.x;
            const __half dy = delta_h2.y;

            const __half h0_5 = __float2half_rn(0.5f);
            const __half dxx = __hmul(dx, dx);
            const __half dyy = __hmul(dy, dy);
            const __half dxy = __hmul(dx, dy);
            const __half quad = __hadd(__hmul(conic_x, dxx), __hmul(conic_z, dyy));
            const __half sigma_over_2_h = __hfma(conic_y, dxy, __hmul(h0_5, quad));
            if (__half2float(sigma_over_2_h) < 0.0f)
                continue;

            const __half gaussian_h = __float2half_rn(__expf(-__half2float(sigma_over_2_h)));
            const __half alpha_raw_h = __hmul(opacity_h, gaussian_h);
            const __half alpha_h = __float2half_rn(fminf(__half2float(alpha_raw_h), config::max_fragment_alpha));
            if (__half2float(alpha_h) < config::min_alpha_threshold)
                continue;

            const __half transmittance_h = color_transmittance_scaled.zw.y;
            const __half next_transmittance_h = __hmul(transmittance_h, __hsub(CUDART_ONE_FP16, alpha_h));
            if (__half2float(next_transmittance_h) < (config::transmittance_threshold * TINYGS_SCALE_FULL)) {
                done = true;
                continue;
            }

            // 颜色累加统一半精度
            const __half2 tah2 = __hmul2(
                __hmul2(make_half2(transmittance_h, transmittance_h), make_half2(alpha_h, alpha_h)),
                TINYGS_UNSCALE_HALF2);
            color_transmittance_scaled.xy = __hfma2(
                tah2,
                make_half2(__ushort2half_rn(collected_color[j].rgb.x),
                           __ushort2half_rn(collected_color[j].rgb.y)),
                color_transmittance_scaled.xy);
            color_transmittance_scaled.zw.x = __hfma(
                tah2.x, __ushort2half_rn(collected_color[j].rgb.z),
                color_transmittance_scaled.zw.x);
            color_transmittance_scaled.zw.y = next_transmittance_h;
            n_contributions = n_possible_contributions;
            if (n_contributions >= config::max_contributions) {
                done = true;
                break;
            }
        }

        j = ((j + 31) / 32) * 32; // round up to next warp
        for (; j < current_batch_size; j += 32) {
            const uint off = tinygs::get_linear_index_tiled(intile.y, intile.x, 2);
            bucket_color_transmittance_scaled[bucket_offset * config::block_size_blend + off] = color_transmittance_scaled;
            bucket_offset++;
        }
    }
    if (inside) {
        const int pixel_idx = width * pixel_coords.y + pixel_coords.x; // logical.
        const uint physical_pixel_idx = tinygs::get_linear_index_tiled(
                /* row */ pixel_coords.y,
                /* col */ pixel_coords.x,
                width_in_tile);

        // Write the buffers, the image is in [0, 1] range.
        color_transmittance_scaled.xy = __hmul2(color_transmittance_scaled.xy, TINYGS_UNSCALE_HALF2);
        color_transmittance_scaled.zw.x = __hmul(color_transmittance_scaled.zw.x, TINYGS_UNSCALE_HALF);
        image[physical_pixel_idx] = color_transmittance_scaled.xy.x;
        image[physical_pixel_idx + channel_stride] = color_transmittance_scaled.xy.y;
        image[physical_pixel_idx + 2 * channel_stride] = color_transmittance_scaled.zw.x;
        alpha_map[physical_pixel_idx] = __hsub(CUDART_ONE_FP16, color_transmittance_scaled.zw.y); // 1-transmittance
        tile_n_contributions[physical_pixel_idx] = static_cast<ushort>(n_contributions);
    }

    // max reduce the number of contributions
    using BlockReduce = cub::BlockReduce<uint, config::tile_width, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_height>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    n_contributions = BlockReduce(temp_storage).Reduce(n_contributions, cub::Max());
    if (thread_rank == 0) {
#ifndef NDEBUG
        // Boundary check for tile arrays
        assert(tile_idx >= 0 && tile_idx < n_tiles);
#endif
        tile_max_n_contributions[tile_idx] = static_cast<ushort>(n_contributions);
    }
}

} // namespace tinygs::fast_gs_fp16::kernels::forward
