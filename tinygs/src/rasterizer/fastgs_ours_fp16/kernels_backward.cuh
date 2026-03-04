/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
// blend_backward based on
// https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404

#pragma once
// Disables `pipeline_shared_state` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init
#include "buffer_utils.h"
#include "../../helper_math.h"
#include "kernel_utils.cuh"
#include "rasterization_config.h"
#include "../sh_soa_utils.cuh"
#include "tinygs/common.hpp"
#include "tinygs/core/gaussian.hpp"
#include "utils.h"
#include <cooperative_groups.h>
#include <cstdint>

#include <cuda/pipeline>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;

namespace tinygs::fast_gs_fp16::kernels::backward {

__device__ __forceinline__ float3 convert_sh_to_color_backward(
    const float* __restrict__ sh1,
    const float* __restrict__ sh2,
    const float* __restrict__ sh3,
    float* __restrict__ grad_sh0,
    float* __restrict__ grad_sh1,
    float* __restrict__ grad_sh2,
    float* __restrict__ grad_sh3,
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
        accum_sh_soa(grad_sh1, 0, N, i, (-0.48860251190291987f * y) * grad_color);
        accum_sh_soa(grad_sh1, 1, N, i, (0.48860251190291987f * z) * grad_color);
        accum_sh_soa(grad_sh1, 2, N, i, (-0.48860251190291987f * x) * grad_color);
        float3 c0 = read_sh_soa(sh1, 0, N, i);
        float3 c1 = read_sh_soa(sh1, 1, N, i);
        float3 c2 = read_sh_soa(sh1, 2, N, i);
        float3 grad_direction_x = -0.48860251190291987f * c2;
        float3 grad_direction_y = -0.48860251190291987f * c0;
        float3 grad_direction_z = 0.48860251190291987f * c1;
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            accum_sh_soa(grad_sh2, 0, N, i, (1.0925484305920792f * xy) * grad_color);
            accum_sh_soa(grad_sh2, 1, N, i, (-1.0925484305920792f * yz) * grad_color);
            accum_sh_soa(grad_sh2, 2, N, i, (0.94617469575755997f * zz - 0.31539156525251999f) * grad_color);
            accum_sh_soa(grad_sh2, 3, N, i, (-1.0925484305920792f * xz) * grad_color);
            accum_sh_soa(grad_sh2, 4, N, i, (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_color);
            float3 c3 = read_sh_soa(sh2, 0, N, i);
            float3 c4 = read_sh_soa(sh2, 1, N, i);
            float3 c5 = read_sh_soa(sh2, 2, N, i);
            float3 c6 = read_sh_soa(sh2, 3, N, i);
            float3 c7 = read_sh_soa(sh2, 4, N, i);
            grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * c3 + (-1.0925484305920792f * z) * c6 + (1.0925484305920792 * x) * c7;
            grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * c3 + (-1.0925484305920792f * z) * c4 + (-1.0925484305920792 * y) * c7;
            grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * c4 + (1.8923493915151202 * z) * c5 + (-1.0925484305920792f * x) * c6;
            if (active_sh_bases > 9) {
                accum_sh_soa(grad_sh3, 0, N, i, (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_color);
                accum_sh_soa(grad_sh3, 1, N, i, (2.8906114426405538f * xy * z) * grad_color);
                accum_sh_soa(grad_sh3, 2, N, i, (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_color);
                accum_sh_soa(grad_sh3, 3, N, i, (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_color);
                accum_sh_soa(grad_sh3, 4, N, i, (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_color);
                accum_sh_soa(grad_sh3, 5, N, i, (1.4453057213202769f * z * (xx - yy)) * grad_color);
                accum_sh_soa(grad_sh3, 6, N, i, (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_color);
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
    const PrimitiveInfoGradient* __restrict__ primitive_info_gradients,
    float3* __restrict__ grad_means,
    float3* __restrict__ grad_raw_scales,
    float4* __restrict__ grad_raw_rotations,
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

    const PrimitiveInfoGradient grad = primitive_info_gradients[primitive_idx];
    // sh evaluation backward
    const float3 primitive_grad_color = make_float3(
        __half2float(grad.color_rg.x),
        __half2float(grad.color_rg.y),
        __half2float(grad.conic_c_color_b.y));
    const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
        sh1, sh2, sh3, grad_sh0, grad_sh1, grad_sh2, grad_sh3,
            primitive_grad_color,
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
        __half2float(grad.conic_ab.x),
        __half2float(grad.conic_ab.y),
        __half2float(grad.conic_c_color_b.x));
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
    // const float2 dL_dmean2d = grad_mean2d[primitive_idx];
    const float2 dL_dmean2d = __half22float2(grad.mean_xy);
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
        const mat3x3 dL_drotation = {
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m11 + rotation_scaled.m21 * dL_dcov3d.m12 + rotation_scaled.m31 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m11 + rotation_scaled.m22 * dL_dcov3d.m12 + rotation_scaled.m32 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m11 + rotation_scaled.m23 * dL_dcov3d.m12 + rotation_scaled.m33 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m12 + rotation_scaled.m21 * dL_dcov3d.m22 + rotation_scaled.m31 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m12 + rotation_scaled.m22 * dL_dcov3d.m22 + rotation_scaled.m32 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m12 + rotation_scaled.m23 * dL_dcov3d.m22 + rotation_scaled.m33 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m13 + rotation_scaled.m21 * dL_dcov3d.m23 + rotation_scaled.m31 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m13 + rotation_scaled.m22 * dL_dcov3d.m23 + rotation_scaled.m32 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m13 + rotation_scaled.m23 * dL_dcov3d.m23 + rotation_scaled.m33 * dL_dcov3d.m33)};
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
        dL_drotation.m12 * (-2.0f * qn_z) + dL_drotation.m13 * ( 2.0f * qn_y) +
        dL_drotation.m21 * ( 2.0f * qn_z) + dL_drotation.m23 * (-2.0f * qn_x) +
        dL_drotation.m31 * (-2.0f * qn_y) + dL_drotation.m32 * ( 2.0f * qn_x);
    const float dL_dqnorm_x =
        dL_drotation.m12 * ( 2.0f * qn_y) + dL_drotation.m13 * ( 2.0f * qn_z) +
        dL_drotation.m21 * ( 2.0f * qn_y) + dL_drotation.m22 * (-4.0f * qn_x) + dL_drotation.m23 * (-2.0f * qn_w) +
        dL_drotation.m31 * ( 2.0f * qn_z) + dL_drotation.m32 * ( 2.0f * qn_w) + dL_drotation.m33 * (-4.0f * qn_x);
    const float dL_dqnorm_y =
        dL_drotation.m11 * (-4.0f * qn_y) + dL_drotation.m12 * ( 2.0f * qn_x) + dL_drotation.m13 * ( 2.0f * qn_w) +
        dL_drotation.m21 * ( 2.0f * qn_x) + dL_drotation.m23 * ( 2.0f * qn_z) +
        dL_drotation.m31 * (-2.0f * qn_w) + dL_drotation.m32 * ( 2.0f * qn_z) + dL_drotation.m33 * (-4.0f * qn_y);
    const float dL_dqnorm_z =
        dL_drotation.m11 * (-4.0f * qn_z) + dL_drotation.m12 * (-2.0f * qn_w) + dL_drotation.m13 * ( 2.0f * qn_x) +
        dL_drotation.m21 * ( 2.0f * qn_w) + dL_drotation.m22 * (-4.0f * qn_z) + dL_drotation.m23 * ( 2.0f * qn_y) +
        dL_drotation.m31 * ( 2.0f * qn_x) + dL_drotation.m32 * ( 2.0f * qn_y);
    
    // Chain through quaternion normalization: q_norm = q_raw / ||q_raw||
    // d(q_norm_i)/d(q_raw_j) = (delta_ij - q_norm_i * q_norm_j) / ||q_raw||
    // dL/d(q_raw) = (dL/d(q_norm) - q_norm * dot(q_norm, dL/d(q_norm))) / ||q_raw||
    const float dot_qnorm_dL = qn_w * dL_dqnorm_w + qn_x * dL_dqnorm_x + qn_y * dL_dqnorm_y + qn_z * dL_dqnorm_z;
    const float4 dL_draw_rotation = make_float4(
        (dL_dqnorm_w - qn_w * dot_qnorm_dL) * inv_q_norm,
        (dL_dqnorm_x - qn_x * dot_qnorm_dL) * inv_q_norm,
        (dL_dqnorm_y - qn_y * dot_qnorm_dL) * inv_q_norm,
        (dL_dqnorm_z - qn_z * dot_qnorm_dL) * inv_q_norm);
    grad_raw_rotations[primitive_idx] += dL_draw_rotation;

        // printf("%d: dL_ddc: %f %f %f\n", 
    //     (int)primitive_idx, grad_sh_coefficients_0[primitive_idx].x, grad_sh_coefficients_0[primitive_idx].y, grad_sh_coefficients_0[primitive_idx].z);

    if (densification_info != nullptr) {
#ifndef NDEBUG
        assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
        densification_info[primitive_idx].accum_counter += 1.0f;
        densification_info[primitive_idx].accum_grad_mean2d += length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));

        const float2 absgrad_mean2d = __half22float2(primitive_info_gradients[primitive_idx].absmean_xy);
        densification_info[primitive_idx].accum_absgrad_mean2d += length(
            absgrad_mean2d * make_float2(0.5f * w, 0.5f * h));
    }
}

// It handles 2 pixels per struct
struct alignas(16) PackedPixels_Upper {
    __half2 grad_color_r;
    __half2 grad_color_g;
    __half2 grad_color_b;
    union {
      ushort2 last_contributor;
      uint32_t last_contributor_ui32;
    };
};

struct alignas(16) PackedPixels_Lower {
    __half2 color_after_r;
    __half2 color_after_g;
    __half2 color_after_b;
    __half2 transmittance;
};


template<typename T>
__device__ __forceinline__ void fast_zero_aligned_8b(T& val) {
    uint64_t* eight_byte = reinterpret_cast<uint64_t*>(&val);
#pragma unroll
    for (int i = 0; i < sizeof(T) / sizeof(uint64_t); ++i) {
        eight_byte[i] = 0;
    }
}

template<typename T>
__device__ __forceinline__ void fast_copy_16bytes(T& dst, const T& src) {
  reinterpret_cast<uint4 &>(dst) = reinterpret_cast<const uint4 &>(src);
}

__device__ __forceinline__ __half2 doth3(__half2 x1, __half2 y1, __half2 z1,
                                     __half2 x2, __half2 y2, __half2 z2) {
    return __hfma2(x1, x2, __hfma2(y1, y2, __hmul2(z1, z2)));
}


__device__ __forceinline__ float sum_float(const __half2& inc) {
    return __half2float(inc.x) + __half2float(inc.y);
}


#ifndef NDEBUG
#define CHECK_FINITE_HALF(v)                                                   \
  do {                                                                         \
    float flt_##v = __half2float(v);                                           \
    assert(isfinite(flt_##v));                                                 \
  } while (0)

#define CHECK_FINITE_HALF2(v)                                                  \
  do {                                                                         \
    float2 flt_##v = __half22float2(v);                                        \
    assert(isfinite(flt_##v.x) && isfinite(flt_##v.y));                        \
  } while (0)

#else

#define CHECK_FINITE_HALF(x) ((void) x)
#define CHECK_FINITE_HALF2(x) ((void) x)

#endif


inline __device__ void atomic_add_gmem_float(float* addr, float in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
	int in_int = *((int*)&in);
	asm ("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(addr), "r"(in_int));
#else
	atomicAdd(addr, in);
#endif
}

inline __device__ void atomic_add_gmem_h2(half2* addr, half2 in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
	int in_int = *((int*)&in);
	asm ("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;" :: "l"(addr), "r"(in_int));
#else
	atomicAdd(addr, in);
#endif
}

/* -------------------- half version -------------------- */
// 2 pixel X 1 GS per thread, cuda driver claim this block size could maximize the occupancy already
__global__ __launch_bounds__(32 * config::blend_bwd_n_warps) void blend_backward_cu2(
    const uint2* __restrict__ tile_instance_ranges,
    const uint* __restrict__ tile_bucket_offsets,
    const uint* __restrict__ instance_primitive_indices,
    const float2* __restrict__ primitive_mean2d,
    const PrimitiveInfo* __restrict__ primitive_info,
    const float16_t* __restrict__ grad_image,
    const float16_t* __restrict__ image,
    const ushort* __restrict__ tile_max_n_contributions,
    const ushort* __restrict__ tile_n_contributions,
    const uint* __restrict__ bucket_tile_index,
    const ColorTransmittance* __restrict__ bucket_color_transmittance_scaled,
    // float2* __restrict__ grad_mean2d,
    // float* __restrict__ grad_conic,
    float* __restrict__ grad_raw_opacity,
    // float3* __restrict__ grad_color,
    PrimitiveInfoGradient* __restrict__ primitive_info_gradients,
    const uint n_buckets,
    const uint n_primitives,
    const uint width,
    const uint height,
    const uint grid_width) {
    constexpr int warp_size = 32;
    constexpr int warp_size_2 = warp_size * 2;
    auto group = cg::this_thread_block();
    auto warp = cg::tiled_partition<warp_size>(group);
    const uint lane_idx = warp.thread_rank();
    const uint warp_idx = group.thread_rank() / warp_size;
    assert(warp_idx < config::blend_bwd_n_warps);

    // -- memcpy async ---
    constexpr uint stages_count = 1;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state[config::blend_bwd_n_warps];
    auto pipeline = cuda::make_pipeline(warp, &shared_state[warp_idx]);

    alignas(16) __shared__ __half2 next_image_r[config::blend_bwd_n_warps][warp_size];
    alignas(16) __shared__ __half2 next_image_g[config::blend_bwd_n_warps][warp_size];
    alignas(16) __shared__ __half2 next_image_b[config::blend_bwd_n_warps][warp_size];
    alignas(16) __shared__ __half2 next_grad_r[config::blend_bwd_n_warps][warp_size];
    alignas(16) __shared__ __half2 next_grad_g[config::blend_bwd_n_warps][warp_size];
    alignas(16) __shared__ __half2 next_grad_b[config::blend_bwd_n_warps][warp_size];
    alignas(16) __shared__ ushort2 next_last_contributor[config::blend_bwd_n_warps][warp_size];
    alignas(16) __shared__ uint4 next_color_transmittance[config::blend_bwd_n_warps][warp_size]; // 16B x 32 = 512 Bytes

    bool is_valid_warp = true;
    uint bucket_idx = (group.group_index().x * config::blend_bwd_n_warps) + warp_idx;
    if (bucket_idx >= n_buckets) {
        is_valid_warp = false;
        // The first warp in each bucket is always valid to read.
        bucket_idx = group.group_index().x * config::blend_bwd_n_warps;
    }
    bucket_color_transmittance_scaled += bucket_idx * config::block_size_blend;

    // tile metadata
    const uint tile_idx = bucket_tile_index[bucket_idx];
    const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
    const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_width};
    const __half2 dist_to_boundaries = make_half2(
       __uint2half_rn(min((uint) (width - start_pixel_coords.x), (uint) config::tile_width)),
       __uint2half_rn(min((uint) (height - start_pixel_coords.y), (uint) config::tile_width))
    );
    const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
    const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
    const uint tile_first_bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
    const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
    if (tile_bucket_idx * 32 >= tile_max_n_contributions[tile_idx]){
      is_valid_warp = false;
    }

    // corresponds to n_contributions
    ushort tile_primitive_idx;
    if (const int tile_primitive_idx_int32 = tile_bucket_idx * 32 + lane_idx;
        tile_primitive_idx_int32 > config::max_contributions) {
      static_assert(((uint)config::max_contributions + 1u) % 32 == 0,
                    "max_contributions + 1 must be divisible by 32 (warp size).");
      is_valid_warp = false;
      // do not return, we need this warp to continue to enable async copies.
    } else {
      // in range => set the variable and continue.
    tile_primitive_idx = (ushort)tile_primitive_idx_int32;
    }
    const uint lane_idx_uint = static_cast<uint>(lane_idx); // thread_idx in the warp, 0 <= lane_idx_uint < 32

    // --- first async copy from gmem ---
    {
        pipeline.producer_acquire();
        const uint width_in_tile = (width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
        const uint height_in_tile = (height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
        const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
        // 0 <= i < 256, since ii < total_pixel_padded < 256 and ii % 64 == 0
        const uint start_pixel_idx = tinygs::get_linear_index_tiled(start_pixel_coords.y, start_pixel_coords.x, width_in_tile);

        // launch all the copies, __half2 = 4B, warp_size = 32 => 128Byte aligned, we cast to 16B is safe
        if (lane_idx_uint < 8) {
            cuda::memcpy_async(reinterpret_cast<uint4*>(next_image_r[warp_idx]) + lane_idx_uint,
                                reinterpret_cast<const uint4*>(image + start_pixel_idx) + lane_idx_uint,
                                cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
            cuda::memcpy_async(reinterpret_cast<uint4*>(next_image_g[warp_idx]) + lane_idx_uint, 
                                reinterpret_cast<const uint4*>(image + start_pixel_idx + channel_stride) + lane_idx_uint,
                                cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
            cuda::memcpy_async(reinterpret_cast<uint4*>(next_image_b[warp_idx]) + lane_idx_uint, 
                                reinterpret_cast<const uint4*>(image + start_pixel_idx + 2 * channel_stride) + lane_idx_uint,
                                cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);

            cuda::memcpy_async(reinterpret_cast<uint4*>(next_grad_r[warp_idx]) + lane_idx_uint,
                                reinterpret_cast<const uint4*>(grad_image + start_pixel_idx) + lane_idx_uint,
                                cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
            cuda::memcpy_async(reinterpret_cast<uint4*>(next_grad_g[warp_idx]) + lane_idx_uint,
                                reinterpret_cast<const uint4*>(grad_image + start_pixel_idx + channel_stride) + lane_idx_uint,
                                cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
            cuda::memcpy_async(reinterpret_cast<uint4*>(next_grad_b[warp_idx]) + lane_idx_uint,
                                reinterpret_cast<const uint4*>(grad_image + start_pixel_idx + 2 * channel_stride) + lane_idx_uint,
                                cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);

            cuda::memcpy_async(reinterpret_cast<uint4*>(next_last_contributor[warp_idx]) + lane_idx_uint,
                                reinterpret_cast<const uint4*>(tile_n_contributions + start_pixel_idx) + lane_idx_uint,
                                cuda::aligned_size_t<16>(sizeof(ushort2) * 4), pipeline);
        }

        cuda::memcpy_async(reinterpret_cast<uint32_t*>(next_color_transmittance[warp_idx] + lane_idx_uint),
                            reinterpret_cast<const uint32_t*>(bucket_color_transmittance_scaled + lane_idx_uint * 2),
                            cuda::aligned_size_t<16>(sizeof(ColorTransmittance) * 2), pipeline);
        pipeline.producer_commit();
    }

    const int instance_idx = tile_instance_range.x + tile_primitive_idx;
    // const bool valid_primitive = tile_primitive_idx < tile_n_primitives && is_valid_warp;
    const uint32_t valid_primitive = (tile_primitive_idx < tile_n_primitives && is_valid_warp) ? 0xFFFF'FFFFu : 0u;

    // --- Constants ---
    const __half2 hinv_16 = __float22half2_rn(make_float2(0.0625f, 0.0625f));
    const __half2 h_16_2 = __float22half2_rn(make_float2(16.0f, 16.0f));
    const __half h0_5 = __float2half_rn(0.5f);
    const __half2 h0_5_2 = make_half2(h0_5, h0_5);
    const __half2 h0_2 = make_half2(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
    const __half2 h_1_2 = make_half2(CUDART_ONE_FP16, CUDART_ONE_FP16);
    const __half2 h_two_pixel_offset_x = make_half2(CUDART_ZERO_FP16, __float2half_rn(1.0f/16.0f));
    const __half2 h_max_fragment_alpha_2 = make_half2(__float2half_rn(config::max_fragment_alpha),
                                                      __float2half_rn(config::max_fragment_alpha));
    const __half2 delta1_offset_x{CUDART_ZERO_FP16, __float2half_rn(1.0f / 16.0f)}; // [0, 1/16]

    // load gaussian data
    uint primitive_idx = 0;
    __half2 mean2d_x{CUDART_ZERO_FP16, CUDART_ZERO_FP16}; // Inf-free
    __half2 mean2d_y{CUDART_ZERO_FP16, CUDART_ZERO_FP16}; // Inf-free
    __half2 conic_x{CUDART_ZERO_FP16, CUDART_ZERO_FP16};  // Inf-free
    __half2 conic_y{CUDART_ZERO_FP16, CUDART_ZERO_FP16};  // Inf-free
    __half2 conic_z{CUDART_ZERO_FP16, CUDART_ZERO_FP16};  // Inf-free
    __half2 opacity{CUDART_ZERO_FP16, CUDART_ZERO_FP16};  // Inf-free
    __half2 color_r{CUDART_ZERO_FP16, CUDART_ZERO_FP16};  // Inf-free
    __half2 color_g{CUDART_ZERO_FP16, CUDART_ZERO_FP16};  // Inf-free
    __half2 color_b{CUDART_ZERO_FP16, CUDART_ZERO_FP16};  // Inf-free
    __half2 color_grad_factor_r{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 color_grad_factor_g{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 color_grad_factor_b{CUDART_ZERO_FP16, CUDART_ZERO_FP16};

    if (valid_primitive) {
        primitive_idx = instance_primitive_indices[instance_idx];
        auto mean2d_float = primitive_mean2d[primitive_idx] / 16.0f - make_float2(tile_coords);
        auto mean2d = __float22half2_rn(mean2d_float - 1.0 / 32.0f);
        mean2d_x = make_half2(mean2d.x, __hsub(mean2d.x, __float2half_rn(1.0f / 16.0f)));
        mean2d_y = make_half2(mean2d.y, mean2d.y);

        const PrimitiveInfo info = primitive_info[primitive_idx];
        conic_x = make_half2(__ushort_as_half(info.conic_xy.x), __ushort_as_half(info.conic_xy.x));
        conic_y = make_half2(__ushort_as_half(info.conic_xy.y), __ushort_as_half(info.conic_xy.y));
        conic_z = make_half2(__ushort_as_half(info.conic_z_opacity.x), __ushort_as_half(info.conic_z_opacity.x));
        opacity = make_half2(__ushort_as_half(info.conic_z_opacity.y), __ushort_as_half(info.conic_z_opacity.y));
        color_r = make_half2(__ushort2half_rn(info.rgb.x), __ushort2half_rn(info.rgb.x));
        color_g = make_half2(__ushort2half_rn(info.rgb.y), __ushort2half_rn(info.rgb.y));
        color_b = make_half2(__ushort2half_rn(info.rgb.z), __ushort2half_rn(info.rgb.z));
        color_grad_factor_r = make_half2(
            (info.rgb.x > 0 && info.rgb.x < 255) ? CUDART_ONE_FP16 : CUDART_ZERO_FP16,
            (info.rgb.x > 0 && info.rgb.x < 255) ? CUDART_ONE_FP16 : CUDART_ZERO_FP16);
        color_grad_factor_g = make_half2(
            (info.rgb.y > 0 && info.rgb.y < 255) ? CUDART_ONE_FP16 : CUDART_ZERO_FP16,
            (info.rgb.y > 0 && info.rgb.y < 255) ? CUDART_ONE_FP16 : CUDART_ZERO_FP16);
        color_grad_factor_b = make_half2(
            (info.rgb.z > 0 && info.rgb.z < 255) ? CUDART_ONE_FP16 : CUDART_ZERO_FP16,
            (info.rgb.z > 0 && info.rgb.z < 255) ? CUDART_ONE_FP16 : CUDART_ZERO_FP16);
    }

    conic_x = __hmul2(conic_x, h_16_2);
    conic_y = __hmul2(conic_y, h_16_2);
    conic_z = __hmul2(conic_z, h_16_2);

    //? Gradient accumulation, kept in float, we are operating one GS's gradients
    //? we do not need half since most half precision operations are about pixels

    __half2 dl_dmean2d_accum_x = h0_2;
    __half2 dl_dmean2d_accum_y = h0_2;
    __half2 abs_dl_dmean2d_accum_x = h0_2;
    __half2 abs_dl_dmean2d_accum_y = h0_2;
    __half2 dl_dconic_accum_x = h0_2;
    __half2 dl_dconic_accum_y = h0_2;
    __half2 dl_dconic_accum_z = h0_2;
    __half2 dl_draw_opacity_partial_accum = h0_2;
    __half2 dl_dcolor_accum_r = h0_2;
    __half2 dl_dcolor_accum_g = h0_2;
    __half2 dl_dcolor_accum_b = h0_2;

    alignas(16) PackedPixels_Upper REGup;   fast_zero_aligned_8b(REGup);
    alignas(16) PackedPixels_Lower REGlow;  fast_zero_aligned_8b(REGlow);

    // One warp is capable of processing 64 pixels at a time in this half version.
    __shared__ PackedPixels_Upper cached_per_pixel_all_upper[config::blend_bwd_n_warps][warp_size];
    __shared__ PackedPixels_Lower cached_per_pixel_all_lower[config::blend_bwd_n_warps][warp_size];

    auto& cached_per_pixel_lower = cached_per_pixel_all_lower[warp_idx];
    auto& cached_per_pixel_upper = cached_per_pixel_all_upper[warp_idx];
    constexpr int total_pixel_padded = config::tile_width * config::tile_width + warp_size_2 - 1;

    __shared__ __half2 cached_off_xy[config::tile_width * config::tile_width + warp_size_2];
    {
        // fill the values in cache_off_xy
        for (uint i = group.thread_rank();
             i < config::tile_width * config::tile_width + warp_size_2;
             i += warp_size * config::blend_bwd_n_warps) {
            if (i >= config::block_size_blend / 2 + warp_size || i < warp_size) {
                cached_off_xy[i] = TINYGS_SCALE_HALF2; // scale = 255, 255, significantly larger than normal values.
            } else {
                const uint idx = (i - warp_size) * 2;
                const uint local_tile = idx >> (2 * tinygs::kImageTileLog2); // 0..3
                const uint intile = idx % (tinygs::kImageTile * tinygs::kImageTile); // 0..63
                const uint dx = (intile % tinygs::kImageTile) + (local_tile % 2) * tinygs::kImageTile; // 0..16
                const uint dy = (intile / tinygs::kImageTile) + (local_tile / 2) * tinygs::kImageTile; // 0..16
                cached_off_xy[i] = make_half2(__ushort2half_rn(dx), __ushort2half_rn(dy));
            }
        }
    }
    group.sync();


    __half2 off_xy{CUDART_MAX_NORMAL_FP16, CUDART_MAX_NORMAL_FP16};
    // iterate over all pixels in the tile
    for (uint ii = 0; ii < total_pixel_padded; ii += warp_size_2) {
        // --- fetch data if not the tail ---
        if (ii < config::block_size_blend) {
            // 0 <= i < 256, since ii < total_pixel_padded < 256 and ii % 64 == 0
            const uint i = ii + lane_idx_uint * 2;
            const uint local_tile = i >> (2 * tinygs::kImageTileLog2);                              // 0..3
            const uint intile = i % (tinygs::kImageTile * tinygs::kImageTile);                      // 0..63
            const uint dx = (intile % tinygs::kImageTile) + (local_tile % 2) * tinygs::kImageTile;  // 0..16
            const uint dy = (intile / tinygs::kImageTile) + (local_tile / 2) * tinygs::kImageTile;  // 0..16
            const uint2 pixel_coords = {start_pixel_coords.x + dx, start_pixel_coords.y + dy};
            const bool is_valid = pixel_coords.x < width && pixel_coords.y < height;
            const bool is_start_valid = (start_pixel_coords.x + (local_tile % 2) * tinygs::kImageTile < width) &&
                                        (start_pixel_coords.y + (local_tile / 2) * tinygs::kImageTile < height);


            PackedPixels_Lower local_lower;
            fast_zero_aligned_8b(local_lower);

            PackedPixels_Upper local_upper;
            fast_zero_aligned_8b(local_upper);

            // Assumes i valid indicates i+1 valid, this is true if the width % 2 == 0, 
            // which is always the case in our application.
            if (is_start_valid){
                pipeline.consumer_wait();
                if (is_valid) {
                    // 1. Load Global Memory
                    local_upper.grad_color_r = next_grad_r[warp_idx][lane_idx_uint];
                    local_upper.grad_color_g = next_grad_g[warp_idx][lane_idx_uint];
                    local_upper.grad_color_b = next_grad_b[warp_idx][lane_idx_uint];
                    local_upper.last_contributor = next_last_contributor[warp_idx][lane_idx_uint];
                    reinterpret_cast<uint4&>(local_lower) = next_color_transmittance[warp_idx][lane_idx_uint];

                    __half2 image_color_r = next_image_r[warp_idx][lane_idx_uint];
                    __half2 image_color_g = next_image_g[warp_idx][lane_idx_uint];
                    __half2 image_color_b = next_image_b[warp_idx][lane_idx_uint];

                    // Compose the results
                    local_lower.color_after_r = __hfma2(image_color_r, TINYGS_SCALE_HALF2, __hneg2(local_lower.color_after_r));
                    local_lower.color_after_g = __hfma2(image_color_g, TINYGS_SCALE_HALF2, __hneg2(local_lower.color_after_g));
                    local_lower.color_after_b = __hfma2(image_color_b, TINYGS_SCALE_HALF2, __hneg2(local_lower.color_after_b));
                    local_lower.transmittance = __hmul2(local_lower.transmittance, TINYGS_UNSCALE_HALF2);
                }
                pipeline.consumer_release();
            }

            // Store to shared
            cached_per_pixel_upper[lane_idx] = local_upper;
            cached_per_pixel_lower[lane_idx] = local_lower;
        }

        // --- async copy from gmem for next tile ---
        if (ii + warp_size_2 < config::block_size_blend) {
            const uint width_in_tile = (width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
            const uint height_in_tile = (height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
            const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
            // 0 <= i < 256, since ii < total_pixel_padded < 256 and ii % 64 == 0
            const uint i = (ii + warp_size_2) ;
            const uint local_tile = i >> (2 * tinygs::kImageTileLog2);                              // 0..3
            const uint dx = (local_tile % 2) * tinygs::kImageTile;  // 0..16
            const uint dy = (local_tile / 2) * tinygs::kImageTile;  // 0..16
            const uint2 pixel_coords = {start_pixel_coords.x + dx, start_pixel_coords.y + dy};
            const uint start_pixel_idx = tinygs::get_linear_index_tiled(
                /* row */ pixel_coords.y, /* col */ pixel_coords.x, width_in_tile);
            const bool is_valid = pixel_coords.x < width && pixel_coords.y < height;
            if (is_valid) {
                pipeline.producer_acquire();
                // launch all the copies, __half2 = 4B, warp_size = 32 => 128Byte aligned, we cast to 16B is safe
                if (lane_idx_uint < 8) {
                    cuda::memcpy_async(reinterpret_cast<uint4*>(next_image_r[warp_idx]) + lane_idx_uint,
                                        reinterpret_cast<const uint4*>(image + start_pixel_idx) + lane_idx_uint,
                                        cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
                    cuda::memcpy_async(reinterpret_cast<uint4*>(next_image_g[warp_idx]) + lane_idx_uint, 
                                        reinterpret_cast<const uint4*>(image + start_pixel_idx + channel_stride) + lane_idx_uint,
                                        cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
                    cuda::memcpy_async(reinterpret_cast<uint4*>(next_image_b[warp_idx]) + lane_idx_uint, 
                                        reinterpret_cast<const uint4*>(image + start_pixel_idx + 2 * channel_stride) + lane_idx_uint,
                                        cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);

                    cuda::memcpy_async(reinterpret_cast<uint4*>(next_grad_r[warp_idx]) + lane_idx_uint,
                                        reinterpret_cast<const uint4*>(grad_image + start_pixel_idx) + lane_idx_uint,
                                        cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
                    cuda::memcpy_async(reinterpret_cast<uint4*>(next_grad_g[warp_idx]) + lane_idx_uint,
                                        reinterpret_cast<const uint4*>(grad_image + start_pixel_idx + channel_stride) + lane_idx_uint,
                                        cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);
                    cuda::memcpy_async(reinterpret_cast<uint4*>(next_grad_b[warp_idx]) + lane_idx_uint,
                                        reinterpret_cast<const uint4*>(grad_image + start_pixel_idx + 2 * channel_stride) + lane_idx_uint,
                                        cuda::aligned_size_t<16>(sizeof(__half2) * 4), pipeline);

                    cuda::memcpy_async(reinterpret_cast<uint4*>(next_last_contributor[warp_idx]) + lane_idx_uint,
                                        reinterpret_cast<const uint4*>(tile_n_contributions + start_pixel_idx) + lane_idx_uint,
                                        cuda::aligned_size_t<16>(sizeof(ushort2) * 4), pipeline);
                }

                cuda::memcpy_async(reinterpret_cast<uint4*>(next_color_transmittance[warp_idx] + lane_idx_uint),
                                    reinterpret_cast<const uint4*>(bucket_color_transmittance_scaled + i + lane_idx_uint * 2),
                                    cuda::aligned_size_t<16>(sizeof(ColorTransmittance) * 2), pipeline);
                pipeline.producer_commit();
            }
        }

        // warp.sync(); // Synchronize after writing to shared memory

        // --- do actural computation ---
        // although the upper bound of j is 32, but deal with 2 pixel per thread/iteration.
#pragma unroll 32
        for (uint j = 0; j < warp_size; ++j) {
            off_xy = cached_off_xy[(ii / 2 + j + warp_size - lane_idx_uint)];

            const __half2 off_x = make_half2(off_xy.x, off_xy.x);
            const __half2 off_y = make_half2(off_xy.y, off_xy.y);
            const uint32_t valid_pixel = __hlt2_mask(off_xy, dist_to_boundaries);
            __half2 delta_x = __hfma2(hinv_16, __hneg2(off_x), mean2d_x);
            __half2 delta_y = __hfma2(hinv_16, __hneg2(off_y), mean2d_y);
            REGup.grad_color_r = warp.shfl_up(REGup.grad_color_r, 1);

            const __half2 conic_x_dx = __hmul2(conic_x, delta_x); // conic.x * delta.x
            const __half2 conic_y_dx = __hmul2(conic_y, delta_x); // conic.y * delta.x
            REGup.grad_color_g = warp.shfl_up(REGup.grad_color_g, 1);
            const __half2 conic_z_dy = __hmul2(conic_z, delta_y); // conic.z * delta.y
            const __half2 conic_y_dy = __hmul2(conic_y, delta_y); // conic.y * delta.y
            REGup.grad_color_b = warp.shfl_up(REGup.grad_color_b, 1);
            uint32_t enable_mask = valid_primitive & valid_pixel;
            const __half2 conic_z_dyy = __hmul2(delta_y, conic_z_dy);
            const __half2 conic_y_dxy = __hmul2(delta_x, conic_y_dy);
            REGup.last_contributor_ui32 = warp.shfl_up(REGup.last_contributor_ui32, 1);
            const uint32_t tile_primitive_idx_u16x2 =
                (static_cast<uint32_t>(tile_primitive_idx) << 16) | static_cast<uint32_t>(tile_primitive_idx);
            enable_mask &= __vcmpltu2(tile_primitive_idx_u16x2, REGup.last_contributor_ui32);

            const __half2 quad = __hfma2(delta_x, conic_x_dx, conic_z_dyy);
            if (lane_idx == 0) {
              fast_copy_16bytes(REGlow, cached_per_pixel_lower[j]);
              fast_copy_16bytes(REGup, cached_per_pixel_upper[j]);
            }
            const __half2 sigma_over_2_h = __hmul2(__hfma2_relu(h0_5_2, quad, conic_y_dxy), h_16_2);
            const __half2 prepare_dl_dmean2d_x = __hadd2(conic_x_dx, conic_y_dy);
            const __half2 gaussian = fast_exp_approx(__hneg2(sigma_over_2_h));
            const __half2 prepare_dl_dmean2d_y = __hadd2(conic_y_dx, conic_z_dy);

            __half2 alpha_prepare = __hmul2(opacity, gaussian);
            reinterpret_cast<uint32_t&>(alpha_prepare) &= enable_mask;
            const __half2 color_dot_grad_color_pixel = doth3(  // scaled by SCALE
                color_r, color_g, color_b,
                REGup.grad_color_r, REGup.grad_color_g, REGup.grad_color_b);

            __half2 alpha = __hmin2(alpha_prepare, h_max_fragment_alpha_2);
            reinterpret_cast<uint32_t&>(delta_x) &= enable_mask; // now ensure it does not has Nan or inf.
            reinterpret_cast<uint32_t&>(delta_y) &= enable_mask; // now ensure it does not has Nan or inf.
            // we have set the maximum transmittance to be about 0.99, and alpha is always larger than half precision.
            const __half2& transmittance = REGlow.transmittance;
            const __half2 blending_weight = __hmul2(transmittance, alpha);
            const __half2 one_minus_alpha = __hsub2(h_1_2, alpha);
            const float2 one_minus_alpha_f = __half22float2(one_minus_alpha);
            const __half2 one_minus_alpha_safe = __float22half2_rn(
                make_float2(
                    fmaxf(one_minus_alpha_f.x, 1e-4f),
                    fmaxf(one_minus_alpha_f.y, 1e-4f)));
            const uint32_t alpha_unsaturated_mask = ~__hge2_mask(alpha_prepare, h_max_fragment_alpha_2);

            // --- color gradient ---
            dl_dcolor_accum_r = __hfma2(__hmul2(blending_weight, color_grad_factor_r), REGup.grad_color_r, dl_dcolor_accum_r);
            dl_dcolor_accum_g = __hfma2(__hmul2(blending_weight, color_grad_factor_g), REGup.grad_color_g, dl_dcolor_accum_g);
            dl_dcolor_accum_b = __hfma2(__hmul2(blending_weight, color_grad_factor_b), REGup.grad_color_b, dl_dcolor_accum_b);

            // --- update reg ---
            // color_pixel_after -= blending_weight * color;
            REGlow.color_after_r = __hfma2(__hneg2(blending_weight), color_r, REGlow.color_after_r);
            REGlow.color_after_g = __hfma2(__hneg2(blending_weight), color_g, REGlow.color_after_g);
            REGlow.color_after_b = __hfma2(__hneg2(blending_weight), color_b, REGlow.color_after_b);

            const __half2 color_pixel_after_dot_grad_color_pixel = doth3(
                REGlow.color_after_r, REGlow.color_after_g, REGlow.color_after_b,
                REGup.grad_color_r, REGup.grad_color_g, REGup.grad_color_b
            );

            // alpha gradient
            // const float dL_dalpha_from_color = transmittance * color_dot_grad_color_pixel - color_pixel_after_dot_grad_color_pixel / one_minus_alpha;
            // const float dL_draw_opacity_partial = alpha * dL_dalpha_from_color;
            // This value does not has NaN of Inf:
            // 1. transmittance is from pixel info,
            // 2. color_dot_grad_color_pixel is safe,
            // 3. alpha has been masked.
            const __half2 dL_dalpha_from_color = __hfma2(transmittance, color_dot_grad_color_pixel,
                                                         __hneg2(__h2div(color_pixel_after_dot_grad_color_pixel, one_minus_alpha_safe)));
            __half2 dL_draw_opacity_partial = __hmul2(alpha, dL_dalpha_from_color);
            reinterpret_cast<uint32_t&>(dL_draw_opacity_partial) &= alpha_unsaturated_mask;

            // dL_draw_opacity_partial_accum += dL_draw_opacity_partial;
            // dL_draw_opacity_partial_accum += sum_float(dL_draw_opacity_partial);
            dl_draw_opacity_partial_accum = __hadd2(dL_draw_opacity_partial, dl_draw_opacity_partial_accum);

            // conic and mean2d gradient
            // const __half2 dL_draw_opacity_partial_neg128 =
            //     __hmul2(dL_draw_opacity_partial, __float22half2_rn(make_float2(-128.f, -128.f)));

            // dL_dconic_accum += dL_dconic;
            const __half2 dxdx = __hmul2(delta_x, delta_x); // if inf is still here, we have dl_draw_opacity_partial == 0.
            const __half2 dxdy = __hmul2(delta_x, delta_y); // if inf is still here, we have dl_draw_opacity_partial == 0.
            const __half2 dydy = __hmul2(delta_y, delta_y); // if inf is still here, we have dl_draw_opacity_partial == 0.
            REGlow.color_after_r = warp.shfl_up(REGlow.color_after_r, 1);
            __half2 dL_dconic_x = __hmul2(dL_draw_opacity_partial, dxdx);
            __half2 dL_dconic_y = __hmul2(dL_draw_opacity_partial, dxdy);
            __half2 dL_dconic_z = __hmul2(dL_draw_opacity_partial, dydy);
            reinterpret_cast<uint32_t&>(dL_dconic_x) &= enable_mask;
            dl_dconic_accum_x = __hadd2(dL_dconic_x, dl_dconic_accum_x);
            REGlow.color_after_g = warp.shfl_up(REGlow.color_after_g, 1);

            reinterpret_cast<uint32_t&>(dL_dconic_y) &= enable_mask;
            dl_dconic_accum_y = __hadd2(dL_dconic_y, dl_dconic_accum_y);

            reinterpret_cast<uint32_t&>(dL_dconic_z) &= enable_mask;
            dl_dconic_accum_z = __hadd2(dL_dconic_z, dl_dconic_accum_z);
            REGlow.color_after_b = warp.shfl_up(REGlow.color_after_b, 1);

            auto new_transmittance = __hmul2(REGlow.transmittance, one_minus_alpha);

            // const float2 dL_dmean2d = dL_draw_opacity_partial * prepare_dl_dmean2d;
            __half2 dL_dmean2d_x = __hmul2(dL_draw_opacity_partial, prepare_dl_dmean2d_x);
            reinterpret_cast<uint32_t&>(dL_dmean2d_x) &= enable_mask;
            __half2 dL_dmean2d_y = __hmul2(dL_draw_opacity_partial, prepare_dl_dmean2d_y);
            reinterpret_cast<uint32_t&>(dL_dmean2d_y) &= enable_mask;

            dl_dmean2d_accum_x = __hadd2(dL_dmean2d_x, dl_dmean2d_accum_x);
            dl_dmean2d_accum_y = __hadd2(dL_dmean2d_y, dl_dmean2d_accum_y);
            const __half2 abs_dldx = __habs2(dL_dmean2d_x);
            const __half2 abs_dldy = __habs2(dL_dmean2d_y);
            abs_dl_dmean2d_accum_x = __hadd2(abs_dldx, abs_dl_dmean2d_accum_x);
            abs_dl_dmean2d_accum_y = __hadd2(abs_dldy, abs_dl_dmean2d_accum_y);

            REGlow.transmittance = warp.shfl_up(new_transmittance, 1);
        }
    }

    // finally add the gradients using atomics
    if (valid_primitive) {

        float2 dL_dmean2d_accum_f = {0.0f, 0.0f};
        float2 absdL_dmean2d_accum_f = {0.0f, 0.0f};
        float3 dL_dconic_accum_f = {0.0f, 0.0f, 0.0f};
        float dL_draw_opacity_partial_accum_f = 0.0f;
        float3 dL_dcolor_accum_f = {0.0f, 0.0f, 0.0f};

        dL_dmean2d_accum_f.x = -sum_float(dl_dmean2d_accum_x) / TINYGS_SCALE_FULL;
        dL_dmean2d_accum_f.y = -sum_float(dl_dmean2d_accum_y) / TINYGS_SCALE_FULL;
        absdL_dmean2d_accum_f.x = sum_float(abs_dl_dmean2d_accum_x) / TINYGS_SCALE_FULL;
        absdL_dmean2d_accum_f.y = sum_float(abs_dl_dmean2d_accum_y) / TINYGS_SCALE_FULL;
        dL_dconic_accum_f.x = - 128.0f * sum_float(dl_dconic_accum_x) / TINYGS_SCALE_FULL;
        dL_dconic_accum_f.y = - 128.0f * sum_float(dl_dconic_accum_y) / TINYGS_SCALE_FULL;
        dL_dconic_accum_f.z = - 128.0f * sum_float(dl_dconic_accum_z) / TINYGS_SCALE_FULL;
        dL_draw_opacity_partial_accum_f = sum_float(dl_draw_opacity_partial_accum) /TINYGS_SCALE_FULL;
        dL_dcolor_accum_f.x = sum_float(dl_dcolor_accum_r);
        dL_dcolor_accum_f.y = sum_float(dl_dcolor_accum_g);
        dL_dcolor_accum_f.z = sum_float(dl_dcolor_accum_b);


#ifndef NDEBUG
        // Boundary check for gradient arrays
        assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
        atomic_add_gmem_h2(&primitive_info_gradients[primitive_idx].mean_xy,
                  __float22half2_rn(make_float2(dL_dmean2d_accum_f.x, dL_dmean2d_accum_f.y)));
        atomic_add_gmem_h2(&primitive_info_gradients[primitive_idx].absmean_xy,
                  __float22half2_rn(make_float2(absdL_dmean2d_accum_f.x, absdL_dmean2d_accum_f.y)));
        const float dL_draw_opacity = dL_draw_opacity_partial_accum_f * (1.0f - __half2float(opacity.x));
        atomic_add_gmem_float(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
        atomic_add_gmem_h2(&primitive_info_gradients[primitive_idx].conic_ab,
                  __float22half2_rn(make_float2(dL_dconic_accum_f.x, dL_dconic_accum_f.y)));
        atomic_add_gmem_h2(&primitive_info_gradients[primitive_idx].color_rg,
                  __float22half2_rn(make_float2(dL_dcolor_accum_f.x, dL_dcolor_accum_f.y)));
        atomic_add_gmem_h2(&primitive_info_gradients[primitive_idx].conic_c_color_b,
                  __float22half2_rn(make_float2(dL_dconic_accum_f.z, dL_dcolor_accum_f.z)));
    }
}

} // namespace tinygs::fast_gs_fp16::kernels::backward
