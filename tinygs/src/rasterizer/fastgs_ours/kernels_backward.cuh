/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "tinygs/core/gaussian.hpp"
// blend_backward based on
// https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404

#include "buffer_utils.h"
#include "../../helper_math.h"
#include "kernel_utils.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include "tinygs/common.hpp"
#include "../sh_soa_utils.cuh"
#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_gs::rasterization::kernels::backward {

/// @brief Backward pass for SH → color, reading/writing SoA buffers directly.
__device__ inline float3 convert_sh_to_color_backward(
    const float* __restrict__ sh1,       // [9*N] band 1 (read)
    const float* __restrict__ sh2,       // [15*N] band 2 (read)
    const float* __restrict__ sh3,       // [21*N] band 3 (read)
    float* __restrict__ grad_sh0,        // [3*N] band 0 gradient (write)
    float* __restrict__ grad_sh1,        // [9*N] band 1 gradient (write)
    float* __restrict__ grad_sh2,        // [15*N] band 2 gradient (write)
    float* __restrict__ grad_sh3,        // [21*N] band 3 gradient (write)
    const float3& grad_color,
    const float3& position,
    const float3& cam_position,
    const uint primitive_idx,
    const uint n_primitives,
    const uint active_sh_bases) {
    using tinygs::read_sh_soa;
    using tinygs::accum_sh0_soa;
    using tinygs::accum_sh_soa;
    const int N = static_cast<int>(n_primitives);
    const int i = static_cast<int>(primitive_idx);
    accum_sh0_soa(grad_sh0, N, i, 0.28209479177387814f * grad_color);
    float3 dcolor_dposition = make_float3(0.0f);
    if (active_sh_bases > 1) {
        auto [x_raw, y_raw, z_raw] = position - cam_position;
        auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
        // Band 1 gradient accumulation
        accum_sh_soa(grad_sh1, 0, N, i, (-0.48860251190291987f * y) * grad_color);
        accum_sh_soa(grad_sh1, 1, N, i, (0.48860251190291987f * z) * grad_color);
        accum_sh_soa(grad_sh1, 2, N, i, (-0.48860251190291987f * x) * grad_color);
        // Read band 1 coefficients for direction gradient
        float3 c0 = read_sh_soa(sh1, 0, N, i);
        float3 c1 = read_sh_soa(sh1, 1, N, i);
        float3 c2 = read_sh_soa(sh1, 2, N, i);
        float3 grad_direction_x = -0.48860251190291987f * c2;
        float3 grad_direction_y = -0.48860251190291987f * c0;
        float3 grad_direction_z = 0.48860251190291987f * c1;
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            // Band 2 gradient accumulation
            accum_sh_soa(grad_sh2, 0, N, i, (1.0925484305920792f * xy) * grad_color);
            accum_sh_soa(grad_sh2, 1, N, i, (-1.0925484305920792f * yz) * grad_color);
            accum_sh_soa(grad_sh2, 2, N, i, (0.94617469575755997f * zz - 0.31539156525251999f) * grad_color);
            accum_sh_soa(grad_sh2, 3, N, i, (-1.0925484305920792f * xz) * grad_color);
            accum_sh_soa(grad_sh2, 4, N, i, (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_color);
            // Read band 2 coefficients for direction gradient
            float3 c3 = read_sh_soa(sh2, 0, N, i);
            float3 c4 = read_sh_soa(sh2, 1, N, i);
            float3 c5 = read_sh_soa(sh2, 2, N, i);
            float3 c6 = read_sh_soa(sh2, 3, N, i);
            float3 c7 = read_sh_soa(sh2, 4, N, i);
            grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * c3 + (-1.0925484305920792f * z) * c6 + (1.0925484305920792 * x) * c7;
            grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * c3 + (-1.0925484305920792f * z) * c4 + (-1.0925484305920792 * y) * c7;
            grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * c4 + (1.8923493915151202 * z) * c5 + (-1.0925484305920792f * x) * c6;
            if (active_sh_bases > 9) {
                // Band 3 gradient accumulation
                accum_sh_soa(grad_sh3, 0, N, i, (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_color);
                accum_sh_soa(grad_sh3, 1, N, i, (2.8906114426405538f * xy * z) * grad_color);
                accum_sh_soa(grad_sh3, 2, N, i, (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_color);
                accum_sh_soa(grad_sh3, 3, N, i, (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_color);
                accum_sh_soa(grad_sh3, 4, N, i, (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_color);
                accum_sh_soa(grad_sh3, 5, N, i, (1.4453057213202769f * z * (xx - yy)) * grad_color);
                accum_sh_soa(grad_sh3, 6, N, i, (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_color);
                // Read band 3 coefficients for direction gradient
                float3 c8  = read_sh_soa(sh3, 0, N, i);
                float3 c9  = read_sh_soa(sh3, 1, N, i);
                float3 c10 = read_sh_soa(sh3, 2, N, i);
                float3 c11 = read_sh_soa(sh3, 3, N, i);
                float3 c12 = read_sh_soa(sh3, 4, N, i);
                float3 c13 = read_sh_soa(sh3, 5, N, i);
                float3 c14 = read_sh_soa(sh3, 6, N, i);
                grad_direction_x = grad_direction_x + (-3.5402615395598609f * xy) * c8 + (2.8906114426405538f * yz) * c9 + (0.45704579946446572f - 2.2852289973223288f * zz) * c12 + (2.8906114426405538f * xz) * c13 + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c14;
                grad_direction_y = grad_direction_y + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * c8 + (2.8906114426405538f * xz) * c9 + (0.45704579946446572f - 2.2852289973223288f * zz) * c10 + (-2.8906114426405538f * yz) * c13 + (3.5402615395598609f * xy) * c14;
                grad_direction_z = grad_direction_z + (2.8906114426405538f * xy) * c9 + (-4.5704579946446566f * yz) * c10 + (5.597644988851731f * zz - 1.1195289977703462f) * c11 + (-4.5704579946446566f * xz) * c12 + (1.4453057213202769f * xx - 1.4453057213202769f * yy) * c13;
            }
        }

        const float3 grad_direction = make_float3(
            dot(grad_direction_x, grad_color),
            dot(grad_direction_y, grad_color),
            dot(grad_direction_z, grad_color));
        const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
        const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
        const float norm_sq = xx_raw + yy_raw + zz_raw;
        dcolor_dposition = make_float3(
                                (yy_raw + zz_raw) * grad_direction.x - xy_raw * grad_direction.y - xz_raw * grad_direction.z,
                                -xy_raw * grad_direction.x + (xx_raw + zz_raw) * grad_direction.y - yz_raw * grad_direction.z,
                                -xz_raw * grad_direction.x - yz_raw * grad_direction.y + (xx_raw + yy_raw) * grad_direction.z) *
                            rsqrtf(norm_sq * norm_sq * norm_sq);
    }
    return dcolor_dposition;
}

__global__ void preprocess_backward_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ raw_scales,
    const float4* __restrict__ raw_rotations,
    const float* __restrict__ sh1,
    const float* __restrict__ sh2,
    const float* __restrict__ sh3,
    const float4* __restrict__ w2c,
    const float3* __restrict__ cam_position,
    const uint* __restrict__ primitive_n_touched_tiles,
    const float2* __restrict__ grad_mean2d,
    const float* __restrict__ grad_conic,
    const float2* __restrict__ absgrad_mean2d,
    float3* __restrict__ grad_means,
    float3* __restrict__ grad_raw_scales,
    float4* __restrict__ grad_raw_rotations,
    float3* __restrict__ grad_color,
    float* __restrict__ grad_sh0,
    float* __restrict__ grad_sh1,
    float* __restrict__ grad_sh2,
    float* __restrict__ grad_sh3,
    float4* __restrict__ grad_w2c_per_gs,
    tinygs::DensificationInfo* __restrict__ densification_info,
    const uint n_primitives,
    const uint active_sh_bases,
    const float w,
    const float h,
    const float fx,
    const float fy,
    const float cx,
    const float cy) {
    auto primitive_idx = cg::this_grid().thread_rank();
    if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0)
        return;

    // load 3d mean
    const float3 mean3d = means[primitive_idx];

    // printf("%d: dl_dsh0: %f %f %f\n", (int)primitive_idx, grad_sh_coefficients_0[primitive_idx].x, grad_sh_coefficients_0[primitive_idx].y, grad_sh_coefficients_0[primitive_idx].z);

    // sh evaluation backward
    const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
        sh1, sh2, sh3, grad_sh0, grad_sh1, grad_sh2, grad_sh3,
        grad_color[primitive_idx],
        mean3d, cam_position[0],
        primitive_idx, n_primitives, active_sh_bases);

    const float4 w2c_r3 = w2c[2];
    const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
    const float4 w2c_r1 = w2c[0];
    const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
    const float4 w2c_r2 = w2c[1];
    const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

    // compute 3d covariance from raw scale and rotation
    const float3 raw_scale = raw_scales[primitive_idx];
    const float3 variance = make_float3(
        tinygs::activate_scale(raw_scale.x) * tinygs::activate_scale(raw_scale.x),
        tinygs::activate_scale(raw_scale.y) * tinygs::activate_scale(raw_scale.y), 
        tinygs::activate_scale(raw_scale.z) * tinygs::activate_scale(raw_scale.z));
    auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
    const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
    const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
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

    // ewa splatting gradient helpers
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

    // 2d covariance gradient
    /// TrickGS: HW / 9Pi N
    // const float dilation = fmaxf(config::dilation, float(h * w) / (9.0f * config::math_pi * n_primitives));
    const float dilation = config::dilation;
    const float a = dot(jwc_r1, jw_r1) + dilation, b = dot(jwc_r1, jw_r2), c = dot(jwc_r2, jw_r2) + dilation;
    const float aa = a * a, bb = b * b, cc = c * c;
    const float ac = a * c, ab = a * b, bc = b * c;
    const float determinant = ac - bb;
    const float determinant_rcp = 1.0f / (determinant + 1e-8f);  // Add epsilon for numerical stability
    const float determinant_rcp_sq = determinant_rcp * determinant_rcp;
    const float3 dL_dconic = make_float3(
        grad_conic[primitive_idx],
        grad_conic[n_primitives + primitive_idx],
        grad_conic[2 * n_primitives + primitive_idx]);
    const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                2.0f * (bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z),
                2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

    // 3d covariance gradient
    const mat3x3_triu dL_dcov3d = {
        (jw_r1.x * jw_r1.x) * dL_dcov2d.x + 2.0f * (jw_r1.x * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.x) * dL_dcov2d.z,
        (jw_r1.x * jw_r1.y) * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.y) * dL_dcov2d.z,
        (jw_r1.x * jw_r1.z) * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + (jw_r2.x * jw_r2.z) * dL_dcov2d.z,
        (jw_r1.y * jw_r1.y) * dL_dcov2d.x + 2.0f * (jw_r1.y * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.y) * dL_dcov2d.z,
        (jw_r1.y * jw_r1.z) * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + (jw_r2.y * jw_r2.z) * dL_dcov2d.z,
        (jw_r1.z * jw_r1.z) * dL_dcov2d.x + 2.0f * (jw_r1.z * jw_r2.z) * dL_dcov2d.y + (jw_r2.z * jw_r2.z) * dL_dcov2d.z,
    };

    // gradient of J * W
    const float3 dL_djw_r1 = 2.0f * make_float3(
                                        jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
                                        jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
                                        jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y);
    const float3 dL_djw_r2 = 2.0f * make_float3(
                                        jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
                                        jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
                                        jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z);

    // gradient of non-zero entries in J
    const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
    const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
    const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
    const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

    // mean3d camera space gradient from J and mean2d
    // Account for clamping of tx/ty in the forward pass. The gradient should only pass if x/y were not clamped.
    const float dtx_dx = (x > clip_left && x < clip_right) ? 1.0f : 0.0f;
    const float dty_dy = (y > clip_top && y < clip_bottom) ? 1.0f : 0.0f;
    const float dL_dj13_clamped = dL_dj13 * dtx_dx;
    const float dL_dj23_clamped = dL_dj23 * dty_dy;

    float djwr1_dz_helper = dL_dj11 - 2.0f * tx * dL_dj13_clamped;
    float djwr2_dz_helper = dL_dj22 - 2.0f * ty * dL_dj23_clamped;
    const float2 dL_dmean2d = grad_mean2d[primitive_idx];
    const float3 dL_dmean3d_cam = make_float3(
        j11 * (dL_dmean2d.x - dL_dj13_clamped / depth),
        j22 * (dL_dmean2d.y - dL_dj23_clamped / depth),
        -j11 * (x * dL_dmean2d.x + djwr1_dz_helper / depth) - j22 * (y * dL_dmean2d.y + djwr2_dz_helper / depth));
    if (grad_w2c_per_gs != nullptr) {
        grad_w2c_per_gs[primitive_idx * 4 + 0].w =  dL_dmean3d_cam.x;
        grad_w2c_per_gs[primitive_idx * 4 + 1].w =  dL_dmean3d_cam.y;
        grad_w2c_per_gs[primitive_idx * 4 + 2].w =  dL_dmean3d_cam.z;
        grad_w2c_per_gs[primitive_idx * 4 + 0].x =  dL_dmean3d_cam.x * mean3d.x;
        grad_w2c_per_gs[primitive_idx * 4 + 0].y =  dL_dmean3d_cam.x * mean3d.y;
        grad_w2c_per_gs[primitive_idx * 4 + 0].z =  dL_dmean3d_cam.x * mean3d.z;
        grad_w2c_per_gs[primitive_idx * 4 + 1].x =  dL_dmean3d_cam.y * mean3d.x;
        grad_w2c_per_gs[primitive_idx * 4 + 1].y =  dL_dmean3d_cam.y * mean3d.y;
        grad_w2c_per_gs[primitive_idx * 4 + 1].z =  dL_dmean3d_cam.y * mean3d.z;
        grad_w2c_per_gs[primitive_idx * 4 + 2].x =  dL_dmean3d_cam.z * mean3d.x;
        grad_w2c_per_gs[primitive_idx * 4 + 2].y =  dL_dmean3d_cam.z * mean3d.y;
        grad_w2c_per_gs[primitive_idx * 4 + 2].z =  dL_dmean3d_cam.z * mean3d.z;
    }

    // 3d mean gradient from splatting
    const float3 dL_dmean3d_from_splatting = make_float3(
        w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
        w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
        w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z);

    // write total 3d mean gradient
    const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
#ifndef NDEBUG
    // Boundary check for primitive arrays
    assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
    grad_means[primitive_idx] += dL_dmean3d;

    // raw scale gradient
    const float dL_dvariance_x = rotation.m11 * rotation.m11 * dL_dcov3d.m11 + rotation.m21 * rotation.m21 * dL_dcov3d.m22 + rotation.m31 * rotation.m31 * dL_dcov3d.m33 +
                                    2.0f * (rotation.m11 * rotation.m21 * dL_dcov3d.m12 + rotation.m11 * rotation.m31 * dL_dcov3d.m13 + rotation.m21 * rotation.m31 * dL_dcov3d.m23);
    const float dL_dvariance_y = rotation.m12 * rotation.m12 * dL_dcov3d.m11 + rotation.m22 * rotation.m22 * dL_dcov3d.m22 + rotation.m32 * rotation.m32 * dL_dcov3d.m33 +
                                    2.0f * (rotation.m12 * rotation.m22 * dL_dcov3d.m12 + rotation.m12 * rotation.m32 * dL_dcov3d.m13 + rotation.m22 * rotation.m32 * dL_dcov3d.m23);
    const float dL_dvariance_z = rotation.m13 * rotation.m13 * dL_dcov3d.m11 + rotation.m23 * rotation.m23 * dL_dcov3d.m22 + rotation.m33 * rotation.m33 * dL_dcov3d.m33 +
                                    2.0f * (rotation.m13 * rotation.m23 * dL_dcov3d.m12 + rotation.m13 * rotation.m33 * dL_dcov3d.m13 + rotation.m23 * rotation.m33 * dL_dcov3d.m23);
    // Original Note:
    // > The gradient for raw_scale is 2*variance*dL_dvariance. When variance is close to zero, this can lead to vanishing gradients.
    // > This is inherent to the exp parameterization of scale, but worth noting for training stability.
    // NOTE: we restore this.
    const float3 dL_draw_scale = make_float3(
        2.0f * tinygs::activate_scale_deriv(raw_scale.x) * tinygs::activate_scale(raw_scale.x) * dL_dvariance_x,
        2.0f * tinygs::activate_scale_deriv(raw_scale.y) * tinygs::activate_scale(raw_scale.y) * dL_dvariance_y,
        2.0f * tinygs::activate_scale_deriv(raw_scale.z) * tinygs::activate_scale(raw_scale.z) * dL_dvariance_z);
#ifndef NDEBUG
    assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
    grad_raw_scales[primitive_idx] += dL_draw_scale;

    // raw rotation gradient
    // dL/d(R) = 2 * dL/d(cov3d) * R * diag(s^2) where R is the unscaled rotation
    // rotation_scaled = R * diag(s^2), so we compute:
    // dL_dR[i][j] = 2 * sum_k dL_dcov3d[i][k] * rotation_scaled[k][j]
    const mat3x3 dL_dR = {
        2.0f * (dL_dcov3d.m11 * rotation_scaled.m11 + dL_dcov3d.m12 * rotation_scaled.m21 + dL_dcov3d.m13 * rotation_scaled.m31),
        2.0f * (dL_dcov3d.m11 * rotation_scaled.m12 + dL_dcov3d.m12 * rotation_scaled.m22 + dL_dcov3d.m13 * rotation_scaled.m32),
        2.0f * (dL_dcov3d.m11 * rotation_scaled.m13 + dL_dcov3d.m12 * rotation_scaled.m23 + dL_dcov3d.m13 * rotation_scaled.m33),
        2.0f * (dL_dcov3d.m12 * rotation_scaled.m11 + dL_dcov3d.m22 * rotation_scaled.m21 + dL_dcov3d.m23 * rotation_scaled.m31),
        2.0f * (dL_dcov3d.m12 * rotation_scaled.m12 + dL_dcov3d.m22 * rotation_scaled.m22 + dL_dcov3d.m23 * rotation_scaled.m32),
        2.0f * (dL_dcov3d.m12 * rotation_scaled.m13 + dL_dcov3d.m22 * rotation_scaled.m23 + dL_dcov3d.m23 * rotation_scaled.m33),
        2.0f * (dL_dcov3d.m13 * rotation_scaled.m11 + dL_dcov3d.m23 * rotation_scaled.m21 + dL_dcov3d.m33 * rotation_scaled.m31),
        2.0f * (dL_dcov3d.m13 * rotation_scaled.m12 + dL_dcov3d.m23 * rotation_scaled.m22 + dL_dcov3d.m33 * rotation_scaled.m32),
        2.0f * (dL_dcov3d.m13 * rotation_scaled.m13 + dL_dcov3d.m23 * rotation_scaled.m23 + dL_dcov3d.m33 * rotation_scaled.m33)};
    
    // Compute dL/d(q_normalized) using the correct derivative formulas for R = mat3_cast(q_norm)
    // The rotation matrix from normalized quaternion (w,x,y,z) is:
    // R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
    //      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
    //      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
    // dR/dw, dR/dx, dR/dy, dR/dz computed via partial derivatives
    const float q_norm = __fsqrt_rn(q_norm_sq);
    const float inv_q_norm = 1.0f / (q_norm + 1e-8f);
    const float qn_w = qr * inv_q_norm;
    const float qn_x = qx * inv_q_norm;
    const float qn_y = qy * inv_q_norm;
    const float qn_z = qz * inv_q_norm;

    const float dL_dqnorm_w = 
        dL_dR.m12 * (-2.0f * qn_z) + dL_dR.m13 * ( 2.0f * qn_y) +
        dL_dR.m21 * ( 2.0f * qn_z) + dL_dR.m23 * (-2.0f * qn_x) +
        dL_dR.m31 * (-2.0f * qn_y) + dL_dR.m32 * ( 2.0f * qn_x);
    const float dL_dqnorm_x =
        dL_dR.m12 * ( 2.0f * qn_y) + dL_dR.m13 * ( 2.0f * qn_z) +
        dL_dR.m21 * ( 2.0f * qn_y) + dL_dR.m22 * (-4.0f * qn_x) + dL_dR.m23 * (-2.0f * qn_w) +
        dL_dR.m31 * ( 2.0f * qn_z) + dL_dR.m32 * ( 2.0f * qn_w) + dL_dR.m33 * (-4.0f * qn_x);
    const float dL_dqnorm_y =
        dL_dR.m11 * (-4.0f * qn_y) + dL_dR.m12 * ( 2.0f * qn_x) + dL_dR.m13 * ( 2.0f * qn_w) +
        dL_dR.m21 * ( 2.0f * qn_x) + dL_dR.m23 * ( 2.0f * qn_z) +
        dL_dR.m31 * (-2.0f * qn_w) + dL_dR.m32 * ( 2.0f * qn_z) + dL_dR.m33 * (-4.0f * qn_y);
    const float dL_dqnorm_z =
        dL_dR.m11 * (-4.0f * qn_z) + dL_dR.m12 * (-2.0f * qn_w) + dL_dR.m13 * ( 2.0f * qn_x) +
        dL_dR.m21 * ( 2.0f * qn_w) + dL_dR.m22 * (-4.0f * qn_z) + dL_dR.m23 * ( 2.0f * qn_y) +
        dL_dR.m31 * ( 2.0f * qn_x) + dL_dR.m32 * ( 2.0f * qn_y);
    
    // Chain through quaternion normalization: q_norm = q_raw / ||q_raw||
    // d(q_norm_i)/d(q_raw_j) = (delta_ij - q_norm_i * q_norm_j) / ||q_raw||
    // dL/d(q_raw) = (dL/d(q_norm) - q_norm * dot(q_norm, dL/d(q_norm))) / ||q_raw||
    const float dot_qnorm_dL = qn_w * dL_dqnorm_w + qn_x * dL_dqnorm_x + qn_y * dL_dqnorm_y + qn_z * dL_dqnorm_z;
    const float4 dL_draw_rotation = make_float4(
        (dL_dqnorm_w - qn_w * dot_qnorm_dL) * inv_q_norm,
        (dL_dqnorm_x - qn_x * dot_qnorm_dL) * inv_q_norm,
        (dL_dqnorm_y - qn_y * dot_qnorm_dL) * inv_q_norm,
        (dL_dqnorm_z - qn_z * dot_qnorm_dL) * inv_q_norm);
#ifndef NDEBUG
    assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
    grad_raw_rotations[primitive_idx] += dL_draw_rotation;

	// printf("%d: dL_ddc: %f %f %f\n", 
    //     (int)primitive_idx, grad_sh_coefficients_0[primitive_idx].x, grad_sh_coefficients_0[primitive_idx].y, grad_sh_coefficients_0[primitive_idx].z);

    if (densification_info != nullptr) {
#ifndef NDEBUG
        assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
        densification_info[primitive_idx].accum_counter += 1.0f;
        densification_info[primitive_idx].accum_grad_mean2d += length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));
        if (absgrad_mean2d != nullptr) {
          densification_info[primitive_idx].accum_absgrad_mean2d += length(
              absgrad_mean2d[primitive_idx] * make_float2(0.5f * w, 0.5f * h));
        }
    }
}

struct alignas(16) PerPixel_Upper {
    float3 grad_color_pixel;
    uint last_contributor;
};

struct alignas(32) PerPixel {
    float3 grad_color_pixel;
    uint last_contributor;
    float3 color_pixel_after;
    float transmittance;
};

struct alignas(16) PerPixel_Lower {
    float3 color_pixel_after;
    float transmittance;
};

static inline __device__ void fast_copy(PerPixel &dst,
                                        const PerPixel &src) {
    uint64_t *dst_ptr = (uint64_t *)&dst;
    const uint64_t *src_ptr = (const uint64_t *)&src;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      dst_ptr[i] = src_ptr[i]; // nvcc will expand all these into two LDS.128 command
    }
}

static inline __device__ void fast_zero(PerPixel &dst) {
    uint64_t *dst_ptr = (uint64_t *)&dst;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      dst_ptr[i] = (uint64_t) 0;
    }
}

static inline __device__ void fast_copy(PerPixel_Upper &dst,
                                        const PerPixel_Upper &src) {
    uint64_t *dst_ptr = (uint64_t *)&dst;
    const uint64_t *src_ptr = (const uint64_t *)&src;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      dst_ptr[i] = src_ptr[i]; // nvcc will expand all these into two LDS.128 command
    }
}

static inline __device__ void fast_copy(PerPixel_Lower &dst,
                                        const PerPixel_Lower &src) {
    uint64_t *dst_ptr = (uint64_t *)&dst;
    const uint64_t *src_ptr = (const uint64_t *)&src;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      dst_ptr[i] = src_ptr[i]; // nvcc will expand all these into two LDS.128 command
    }
}

static inline __device__ void fast_zero(PerPixel_Upper &dst) {
        uint64_t *dst_ptr = (uint64_t *)&dst;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            dst_ptr[i] = (uint64_t) 0;
        }
}

static inline __device__ void fast_zero(PerPixel_Lower &dst) {
        uint64_t *dst_ptr = (uint64_t *)&dst;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            dst_ptr[i] = (uint64_t) 0;
        }
}
__global__ __launch_bounds__(32 * config::blend_bwd_n_warps) void blend_backward_cu2(
    const uint2* __restrict__ tile_instance_ranges,
    const uint* __restrict__ tile_bucket_offsets,
    const uint* __restrict__ instance_primitive_indices,
    const float2* __restrict__ primitive_mean2d,
    const float4* __restrict__ primitive_conic_opacity,
    const float3* __restrict__ primitive_color,
    const float* __restrict__ grad_image,
    const float* __restrict__ image,
    const uint* __restrict__ tile_max_n_contributions,
    const uint* __restrict__ tile_n_contributions,
    const uint* __restrict__ bucket_tile_index,
    const float4* __restrict__ bucket_color_transmittance,
    float2* __restrict__ grad_mean2d,
    float2* __restrict__ absgrad_mean2d,
    float* __restrict__ grad_conic,
    float* __restrict__ grad_raw_opacity,
    float3* __restrict__ grad_color,
    const uint n_buckets,
    const uint n_primitives,
    const uint width,
    const uint height,
    const uint grid_width) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint lane_idx = warp.thread_rank();
    const uint warp_idx = block.thread_rank() / 32;

    assert(warp_idx < config::blend_bwd_n_warps);
    const uint bucket_idx = (block.group_index().x * config::blend_bwd_n_warps) + warp_idx;

    if (bucket_idx >= n_buckets)
        return;

    const uint tile_idx = bucket_tile_index[bucket_idx];
    const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
    const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
    const uint tile_first_bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
    const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
    if (tile_bucket_idx * 32 >= tile_max_n_contributions[tile_idx])
        return;

    const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
    const int instance_idx = tile_instance_range.x + tile_primitive_idx;
    const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

    // load gaussian data
    uint primitive_idx = 0;
    float2 mean2d = {0.0f, 0.0f};
    float3 conic = {0.0f, 0.0f, 0.0f};
    float opacity = 0.0f;
    float3 color = {0.0f, 0.0f, 0.0f};
    float3 color_grad_factor = {0.0f, 0.0f, 0.0f};

    // tile metadata
    const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
    const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

    bucket_color_transmittance += bucket_idx * config::block_size_blend;

    if (valid_primitive) {
        primitive_idx = instance_primitive_indices[instance_idx];
        mean2d = primitive_mean2d[primitive_idx];
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        conic = make_float3(conic_opacity);
        opacity = conic_opacity.w;
        const float3 color_unclamped = primitive_color[primitive_idx];
        color = fmaxf(color_unclamped, 0.0f);
        if (color_unclamped.x >= 0.0f)
            color_grad_factor.x = 1.0f;
        if (color_unclamped.y >= 0.0f)
            color_grad_factor.y = 1.0f;
        if (color_unclamped.z >= 0.0f)
            color_grad_factor.z = 1.0f;
    }


    // gradient accumulation
    float2 dL_dmean2d_accum = {0.0f, 0.0f};
    float2 absdL_dmean2d_accum = {0.0f, 0.0f};
    float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
    float dL_draw_opacity_partial_accum = 0.0f;
    float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

    alignas(32) PerPixel per_pixel_registers;
    fast_zero(per_pixel_registers);

    // shorter
    auto& last_contributor = per_pixel_registers.last_contributor;
    auto& color_pixel_after = per_pixel_registers.color_pixel_after;
    auto& transmittance = per_pixel_registers.transmittance;
    auto& grad_color_pixel = per_pixel_registers.grad_color_pixel;


    __shared__ PerPixel_Upper cached_per_pixel_all_upper[config::blend_bwd_n_warps][32];
    __shared__ PerPixel_Lower cached_per_pixel_all_lower[config::blend_bwd_n_warps][32];
    auto& cached_per_pixel_lower = cached_per_pixel_all_lower[warp_idx];
    auto& cached_per_pixel_upper = cached_per_pixel_all_upper[warp_idx];
    const uint lane_idx_uint = static_cast<uint>(lane_idx);
    unsigned long long saddr_lower, saddr_upper;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(saddr_lower) : "l"(cached_per_pixel_lower));
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(saddr_upper) : "l"(cached_per_pixel_upper));

    // iterate over all pixels in the tile
    for (uint ii = 0; ii < config::block_size_blend + 31; ii += 32) {
        if (ii < config::block_size_blend) { // fetch data
            const uint width_in_tile = (width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
            const uint height_in_tile = (height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
            const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
            const uint i = ii + lane_idx_uint; // 0 <= i < 256
            const uint local_tile = i >> (2 * tinygs::kImageTileLog2);                              // 0..3
            const uint intile = i % (tinygs::kImageTile * tinygs::kImageTile);                      // 0..63
            const uint dx = (intile % tinygs::kImageTile) + (local_tile % 2) * tinygs::kImageTile;  // 0..16
            const uint dy = (intile / tinygs::kImageTile) + (local_tile / 2) * tinygs::kImageTile;  // 0..16
            assert(local_tile < 4);
            assert(intile < tinygs::kImageTile * tinygs::kImageTile);
            assert(dx < config::tile_width);
            assert(dy < config::tile_height);
            const uint2 pixel_coords = {start_pixel_coords.x + dx,
                                        start_pixel_coords.y + dy};
            // const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const uint physical_pixel_idx = tinygs::get_linear_index_tiled(
                /* row */ pixel_coords.y,
                /* col */ pixel_coords.x,
                width_in_tile);
            const bool is_valid =
                pixel_coords.x < width && pixel_coords.y < height &&
                dx < config::tile_width && dy < config::tile_height;
            PerPixel_Lower local_lower;
            fast_zero(local_lower);

            PerPixel_Upper local_upper;
            fast_zero(local_upper);
            float4 color_transmittance{0.f, 0.f, 0.f, 0.f};

            if (is_valid) {
                color_transmittance = bucket_color_transmittance[i];
                local_upper.last_contributor = tile_n_contributions[physical_pixel_idx];
                local_upper.grad_color_pixel = make_float3(grad_image[physical_pixel_idx],
                                grad_image[physical_pixel_idx + channel_stride],
                                grad_image[physical_pixel_idx + channel_stride * 2]);
                local_lower.color_pixel_after = make_float3(image[physical_pixel_idx],
                                image[physical_pixel_idx + channel_stride],
                                image[physical_pixel_idx + channel_stride * 2]);
                local_lower.transmittance = color_transmittance.w;
            }
            local_lower.color_pixel_after = local_lower.color_pixel_after - make_float3(color_transmittance);
            fast_copy(cached_per_pixel_lower[lane_idx], local_lower);
            fast_copy(cached_per_pixel_upper[lane_idx], local_upper);
            __syncwarp(); // Synchronize after writing to shared memory
        }

#pragma unroll
        for (uint j = 0; j < 32; ++j) {
            const uint i = ii + j;
            // which pixel index should this thread deal with?
            const uint idx = i - lane_idx_uint; // overflow is ok, will much greater than the block size, and mark invalid
            const uint local_tile = idx >> (2 * tinygs::kImageTileLog2); // 0..3
            const uint intile = idx % (tinygs::kImageTile * tinygs::kImageTile); // 0..63
            const uint dx = (intile % tinygs::kImageTile) + (local_tile % 2) * tinygs::kImageTile; // 0..16
            const uint dy = (intile / tinygs::kImageTile) + (local_tile / 2) * tinygs::kImageTile; // 0..16
            const uint2 pixel_coords = {
                start_pixel_coords.x + dx,
                start_pixel_coords.y + dy};
            per_pixel_registers = warp.shfl_up(per_pixel_registers, 1);

            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;
            const bool valid_general = valid_primitive && valid_pixel && idx < config::block_size_blend;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y));
            const float2 delta = (mean2d - 0.5f) - pixel;
            const float3 delta_coefs = make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
            const float sigma_over_2_gt = 0.5f * (conic.x * delta_coefs.x + conic.z * delta_coefs.z) + conic.y * delta_coefs.y;
            const float sigma_over_2 = fmaxf(sigma_over_2_gt, 0.0f); // ensures >= 0
            const float gaussian = __expf(-sigma_over_2);

            // leader thread loads values from shared memory into registers
            if (lane_idx == 0 && valid_general) {
                float4* dst_view = reinterpret_cast<float4*>(&per_pixel_registers);
                float4* dst_view_next = dst_view + 1;
                // asm this. fuck
                asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(dst_view->x), "=f"(dst_view->y), "=f"(dst_view->z), "=f"(dst_view->w)
                    : "l"(saddr_upper + (i % 32) * sizeof(PerPixel_Upper)));
                asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                    : "=f"(dst_view_next->x), "=f"(dst_view_next->y), "=f"(dst_view_next->z), "=f"(dst_view_next->w)
                    : "l"(saddr_lower + (i % 32) * sizeof(PerPixel_Lower)));
            }
            __syncwarp(); // Synchronize after reading from shared memory
            const bool skip = !valid_general || tile_primitive_idx >= last_contributor;
            const float alpha_prepare = opacity * gaussian;
            const float color_dot_grad_color_pixel = dot(color, grad_color_pixel);

            float alpha = 0.f;
            if (!skip) [[likely]] {
                alpha = fminf(alpha_prepare, config::max_fragment_alpha);
            }
            const bool alpha_saturated = alpha_prepare >= config::max_fragment_alpha;

            const float blending_weight = transmittance * alpha;
            // const float inv_contribution = sqrtf(1.0f / (blending_weight + config::min_alpha_threshold));
            const float inv_contribution = 1;
            const float one_minus_alpha = 1.0f - alpha;
            const float one_minus_alpha_safe = fmaxf(one_minus_alpha, 1e-4f);
            const float one_minus_alpha_rcp = 1.0f / one_minus_alpha_safe;
            // color gradient
            const float3 dL_dcolor = blending_weight * (grad_color_pixel * color_grad_factor);
            // dL_dcolor_accum += dL_dcolor;
            dL_dcolor_accum += dL_dcolor * inv_contribution;
            color_pixel_after -= blending_weight * color;
            const float color_pixel_after_dot_grad_color_pixel = dot(color_pixel_after, grad_color_pixel);
            const float2 prepare_dl_dmean2d =
                make_float2(conic.x * delta.x + conic.y * delta.y,
                            conic.y * delta.x + conic.z * delta.y);

            // alpha gradient
            const float dL_dalpha_from_color = transmittance * color_dot_grad_color_pixel - color_pixel_after_dot_grad_color_pixel * one_minus_alpha_rcp;
            const float dL_draw_opacity_partial = alpha_saturated ? 0.0f : alpha * dL_dalpha_from_color;
            // dL_draw_opacity_partial_accum += dL_draw_opacity_partial;
            dL_draw_opacity_partial_accum += dL_draw_opacity_partial * inv_contribution;

            // conic and mean2d gradient
            const float3 dL_dconic = -0.5f * dL_draw_opacity_partial * delta_coefs;
            // dL_dconic_accum += dL_dconic;
            dL_dconic_accum += dL_dconic * inv_contribution;
            const float2 dL_dmean2d = dL_draw_opacity_partial * prepare_dl_dmean2d;

            // dL_dmean2d_accum -= dL_dmean2d;
            dL_dmean2d_accum -= dL_dmean2d * inv_contribution;
            absdL_dmean2d_accum += make_float2(fabsf(dL_dmean2d.x), fabsf(dL_dmean2d.y));
            transmittance *= one_minus_alpha;
        }
    }

    // finally add the gradients using atomics
    if (valid_primitive) {
#ifndef NDEBUG
        // Boundary check for gradient arrays
        assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
        atomicAdd(&grad_mean2d[primitive_idx].x, dL_dmean2d_accum.x);
        atomicAdd(&grad_mean2d[primitive_idx].y, dL_dmean2d_accum.y);
        if (absgrad_mean2d != nullptr) {
            atomicAdd(&absgrad_mean2d[primitive_idx].x, absdL_dmean2d_accum.x);
            atomicAdd(&absgrad_mean2d[primitive_idx].y, absdL_dmean2d_accum.y);
        }
        atomicAdd(&grad_conic[primitive_idx], dL_dconic_accum.x);
        atomicAdd(&grad_conic[n_primitives + primitive_idx], dL_dconic_accum.y);
        atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], dL_dconic_accum.z);
        const float dL_draw_opacity = dL_draw_opacity_partial_accum * (1.0f - opacity);
        atomicAdd(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
        atomicAdd(&grad_color[primitive_idx].x, dL_dcolor_accum.x);
        atomicAdd(&grad_color[primitive_idx].y, dL_dcolor_accum.y);
        atomicAdd(&grad_color[primitive_idx].z, dL_dcolor_accum.z);
    }
}


} // namespace fast_gs::rasterization::kernels::backward
