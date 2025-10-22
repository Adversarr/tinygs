/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "tinygs/core/gaussian.hpp"
// blend_backward based on
// https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include "tinygs/common.hpp"
#include <cooperative_groups.h>
#include <cstdint>

#include <cuda_bf16.h>

namespace cg = cooperative_groups;

namespace tinygs::fast_gs_fp16::kernels::backward {

__device__ inline float3 convert_sh_to_color_backward(
    const float3* sh_coefficients_rest,
    float3* grad_sh_coefficients_0,
    float3* grad_sh_coefficients_rest,
    const float3& grad_color,
    const float3& position,
    const float3& cam_position,
    const uint primitive_idx,
    const uint active_sh_bases,
    const uint total_bases_sh_rest) {
    // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    const int coefficients_base_idx = primitive_idx * total_bases_sh_rest;
    const float3* coefficients_ptr = sh_coefficients_rest + coefficients_base_idx;
    float3* grad_coefficients_ptr = grad_sh_coefficients_rest + coefficients_base_idx;
    grad_sh_coefficients_0[primitive_idx] += 0.28209479177387814f * grad_color;
    float3 dcolor_dposition = make_float3(0.0f);
    if (active_sh_bases > 1) {
        auto [x_raw, y_raw, z_raw] = position - cam_position;
        auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
        grad_coefficients_ptr[0] += (-0.48860251190291987f * y) * grad_color;
        grad_coefficients_ptr[1] += (0.48860251190291987f * z) * grad_color;
        grad_coefficients_ptr[2] += (-0.48860251190291987f * x) * grad_color;
        float3 grad_direction_x = -0.48860251190291987f * coefficients_ptr[2];
        float3 grad_direction_y = -0.48860251190291987f * coefficients_ptr[0];
        float3 grad_direction_z = 0.48860251190291987f * coefficients_ptr[1];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            grad_coefficients_ptr[3] += (1.0925484305920792f * xy) * grad_color;
            grad_coefficients_ptr[4] += (-1.0925484305920792f * yz) * grad_color;
            grad_coefficients_ptr[5] += (0.94617469575755997f * zz - 0.31539156525251999f) * grad_color;
            grad_coefficients_ptr[6] += (-1.0925484305920792f * xz) * grad_color;
            grad_coefficients_ptr[7] += (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_color;
            grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * coefficients_ptr[3] + (-1.0925484305920792f * z) * coefficients_ptr[6] + (1.0925484305920792 * x) * coefficients_ptr[7];
            grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * coefficients_ptr[3] + (-1.0925484305920792f * z) * coefficients_ptr[4] + (-1.0925484305920792 * y) * coefficients_ptr[7];
            grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * coefficients_ptr[4] + (1.8923493915151202 * z) * coefficients_ptr[5] + (-1.0925484305920792f * x) * coefficients_ptr[6];
            if (active_sh_bases > 9) {
                grad_coefficients_ptr[8] += (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_color;
                grad_coefficients_ptr[9] += (2.8906114426405538f * xy * z) * grad_color;
                grad_coefficients_ptr[10] += (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_color;
                grad_coefficients_ptr[11] += (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_color;
                grad_coefficients_ptr[12] += (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_color;
                grad_coefficients_ptr[13] += (1.4453057213202769f * z * (xx - yy)) * grad_color;
                grad_coefficients_ptr[14] += (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_color;
                grad_direction_x = grad_direction_x + (-3.5402615395598609f * xy) * coefficients_ptr[8] + (2.8906114426405538f * yz) * coefficients_ptr[9] + (0.45704579946446572f - 2.2852289973223288f * zz) * coefficients_ptr[12] + (2.8906114426405538f * xz) * coefficients_ptr[13] + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * coefficients_ptr[14];
                grad_direction_y = grad_direction_y + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * coefficients_ptr[8] + (2.8906114426405538f * xz) * coefficients_ptr[9] + (0.45704579946446572f - 2.2852289973223288f * zz) * coefficients_ptr[10] + (-2.8906114426405538f * yz) * coefficients_ptr[13] + (3.5402615395598609f * xy) * coefficients_ptr[14];
                grad_direction_z = grad_direction_z + (2.8906114426405538f * xy) * coefficients_ptr[9] + (-4.5704579946446566f * yz) * coefficients_ptr[10] + (5.597644988851731f * zz - 1.1195289977703462f) * coefficients_ptr[11] + (-4.5704579946446566f * xz) * coefficients_ptr[12] + (1.4453057213202769f * xx - 1.4453057213202769f * yy) * coefficients_ptr[13];
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
    const float3* __restrict__ sh_coefficients_rest,
    const float4* __restrict__ w2c,
    const float3* __restrict__ cam_position,
    const uint* __restrict__ primitive_n_touched_tiles,
    // const float2* __restrict__ grad_mean2d,
    // const float* __restrict__ grad_conic,
    const PrimitiveInfoGradient* __restrict__ primitive_info_gradients,
    const float2* __restrict__ absgrad_mean2d,
    float3* __restrict__ grad_means,
    float3* __restrict__ grad_raw_scales,
    float4* __restrict__ grad_raw_rotations,
    // float3* __restrict__ grad_color,
    float3* __restrict__ grad_sh_coefficients_0,
    float3* __restrict__ grad_sh_coefficients_rest,
    float4* __restrict__ grad_w2c_per_gs,
    tinygs::DensificationInfo* __restrict__ densification_info,
    const uint n_primitives,
    const uint active_sh_bases,
    const uint total_bases_sh_rest,
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
    const float3 primitive_grad_color = make_float3(
        __half2float(primitive_info_gradients[primitive_idx].color_rg.x),
        __half2float(primitive_info_gradients[primitive_idx].color_rg.y),
        __half2float(primitive_info_gradients[primitive_idx].conic_c_color_b.y));
    const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
        sh_coefficients_rest, grad_sh_coefficients_0, grad_sh_coefficients_rest,
        primitive_grad_color,
        mean3d, cam_position[0],
        primitive_idx, active_sh_bases, total_bases_sh_rest);

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
        __half2float(primitive_info_gradients[primitive_idx].conic_ab.x),
        __half2float(primitive_info_gradients[primitive_idx].conic_ab.y),
        __half2float(primitive_info_gradients[primitive_idx].conic_c_color_b.x));
    const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
                2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
                // GPT-5 claims here should have a 2.0f, but the reference does not have it
                /* 2.0f * */ (bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z),
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
    const float2 dL_dmean2d = __half22float2(primitive_info_gradients[primitive_idx].mean_xy);
    const float3 dL_dmean3d_cam = make_float3(
        j11 * (dL_dmean2d.x - dL_dj13_clamped / depth),
        j22 * (dL_dmean2d.y - dL_dj23_clamped / depth),
        -j11 * (x * dL_dmean2d.x + djwr1_dz_helper / depth) - j22 * (y * dL_dmean2d.y + djwr2_dz_helper / depth));

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
    const float dL_dqxx = -dL_drotation.m22 - dL_drotation.m33;
    const float dL_dqyy = -dL_drotation.m11 - dL_drotation.m33;
    const float dL_dqzz = -dL_drotation.m11 - dL_drotation.m22;
    const float dL_dqxy = dL_drotation.m12 + dL_drotation.m21;
    const float dL_dqxz = dL_drotation.m13 + dL_drotation.m31;
    const float dL_dqyz = dL_drotation.m23 + dL_drotation.m32;
    const float dL_dqrx = dL_drotation.m32 - dL_drotation.m23;
    const float dL_dqry = dL_drotation.m13 - dL_drotation.m31;
    const float dL_dqrz = dL_drotation.m21 - dL_drotation.m12;
    // The following formula for quaternion gradient appears to be a custom implementation.
    // It's recommended to verify its correctness against the original 3DGS paper or standard quaternion calculus references.
    const float dL_dq_norm_helper = qxx * dL_dqxx + qyy * dL_dqyy + qzz * dL_dqzz + qxy * dL_dqxy + qxz * dL_dqxz + qyz * dL_dqyz + qrx * dL_dqrx + qry * dL_dqry + qrz * dL_dqrz;
    const float4 dL_draw_rotation = 2.0f * make_float4(
        qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper,
        2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper,
        2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper,
        2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) / (q_norm_sq * __fsqrt_rn(q_norm_sq));
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

struct alignas(8) PerPixel_Upper {
    __half2 grad_color_pixel_rg;
    __half2_raw grad_color_pixel_b_last_contributor;
};

using PerPixel_Lower = packed_half2x2;

struct alignas(16) PerPixel {
    __half2 grad_color_pixel_rg;
    __half2_raw grad_color_pixel_b_last_contributor;
    __half2 color_pixel_after_rg;
    __half2 color_pixel_after_b_transmittance;
};

static inline __device__ void fast_zero(PerPixel &dst) {
    uint64_t *dst_ptr = (uint64_t *)&dst;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      dst_ptr[i] = (uint64_t) 0;
    }
}

static inline __device__ void fast_copy(PerPixel_Upper &dst,
                                        const PerPixel_Upper &src) {
    uint64_t *dst_ptr = (uint64_t *)&dst;
    const uint64_t *src_ptr = (const uint64_t *)&src;
#pragma unroll
    for (int i = 0; i < 1; i++) {
      dst_ptr[i] = src_ptr[i]; // nvcc will expand all these into two LDS.128 command
    }
}

static inline __device__ void fast_zero(PerPixel_Upper &dst) {
    reinterpret_cast<uint64_t&>(dst) = 0ull;
}
__device__ __half dot3(const packed_half2x2 &a, const packed_half2x2 &b)
{
    // a.xy*b.xy + 0
    __half2 accum = __hmul2(a.xy, b.xy);
    // + a.zw.x*b.zw.x
    accum.x = __hfma(a.zw.x, b.zw.x, accum.x);
    // now, accum = (a.x*b.x + a.z*b.z, a.y*b.y)
    __half res = __hadd(__low2half(accum), __high2half(accum));
    return res;
}

__global__ __launch_bounds__(32 * config::blend_bwd_n_warps) void blend_backward_cu(
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
    float2* __restrict__ absgrad_mean2d,
    // float* __restrict__ grad_conic,
    float* __restrict__ grad_raw_opacity,
    // float3* __restrict__ grad_color,
    PrimitiveInfoGradient* __restrict__ primitive_info_gradients,
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

    // corresponds to n_contributions
    ushort tile_primitive_idx;
    if (const int tile_primitive_idx_int32 = tile_bucket_idx * 32 + lane_idx;
        tile_primitive_idx_int32 > config::max_contributions) {
      static_assert(((uint)config::max_contributions + 1u) % 32 == 0,
                    "max_contributions + 1 must be divisible by 32 (warp size).");
      // out of range, skip, it is safe due to max_contributions + 1 is
      // divisible by 32
      return;
    } else {
      // in range => set the variable and continue.
      tile_primitive_idx = (ushort)tile_primitive_idx_int32;
    }

    const int instance_idx = tile_instance_range.x + tile_primitive_idx;
    const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

    // load gaussian data
    uint primitive_idx = 0;
    float2 mean2d = {0.0f, 0.0f};
    float3 conic = {0.0f, 0.0f, 0.0f};
    float opacity = 0.0f;
    packed_half2x2 color;
    fast_zero(color);

    // tile metadata
    const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
    const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_width};

    bucket_color_transmittance_scaled += bucket_idx * config::block_size_blend;

    if (valid_primitive) {
        primitive_idx = instance_primitive_indices[instance_idx];
        mean2d = primitive_mean2d[primitive_idx] / 16.0f - make_float2(tile_coords);
        const auto info = primitive_info[primitive_idx];
        conic = make_float3(
            __half2float(__ushort_as_half(info.conic_xy.x)),
            __half2float(__ushort_as_half(info.conic_xy.y)),
            __half2float(__ushort_as_half(info.conic_z_raw_opacity.x)));
        opacity = activate_opacity(__half2float(__ushort_as_half(info.conic_z_raw_opacity.y)));
        color.xy = __hmul2(make_half2(__ushort2half_rn(info.rgb.x), __ushort2half_rn(info.rgb.y)),
                           TINYGS_UNSCALE_HALF2);
        color.zw.x = __hmul(__ushort2half_rn(info.rgb.z), TINYGS_UNSCALE_HALF);
    }


    // gradient accumulation
    float2 dL_dmean2d_accum = {0.0f, 0.0f};
    float2 absdL_dmean2d_accum = {0.0f, 0.0f};
    float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
    float dL_draw_opacity_partial_accum = 0.0f;
    float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

    union union_per_pixel {
      PerPixel full; // 完整 16B 结构
      struct {
        PerPixel_Upper upper; // 前 8B
        PerPixel_Lower lower; // 后 8B
      } parts;

      uint4 as_uint4; // also 16B
    } REG;
    fast_zero(REG.full);

    // shorter
    // auto& last_contributor = per_pixel_registers.last_contributor;
    // auto& color_pixel_after = per_pixel_registers.color_pixel_after;
    // auto& transmittance = per_pixel_registers.transmittance;
    // auto& grad_color_pixel = per_pixel_registers.grad_color_pixel;


    __shared__ PerPixel_Upper cached_per_pixel_all_upper[config::blend_bwd_n_warps][32];
    __shared__ PerPixel_Lower cached_per_pixel_all_lower[config::blend_bwd_n_warps][32];
    auto& cached_per_pixel_lower = cached_per_pixel_all_lower[warp_idx];
    auto& cached_per_pixel_upper = cached_per_pixel_all_upper[warp_idx];
    const uint lane_idx_uint = static_cast<uint>(lane_idx);
    unsigned long long saddr_lower, saddr_upper;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(saddr_lower) : "l"(cached_per_pixel_lower));
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(saddr_upper) : "l"(cached_per_pixel_upper));


    // --- constants ---
    const __half h16 = __float2half_rn(16.0f);
    const __half h0_5 = __float2half_rn(0.5f);

    // iterate over all pixels in the tile
    for (uint ii = 0; ii < config::block_size_blend + 31; ii += 32) {
        if (ii < config::block_size_blend) {  // fetch data
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
            assert(dy < config::tile_width);
            const uint2 pixel_coords = {start_pixel_coords.x + dx, start_pixel_coords.y + dy};

            // const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const uint physical_pixel_idx = tinygs::get_linear_index_tiled(
                /* row */ pixel_coords.y,
                /* col */ pixel_coords.x,
                width_in_tile);
            const bool is_valid =
                pixel_coords.x < width && pixel_coords.y < height &&
                dx < config::tile_width && dy < config::tile_width;

            PerPixel_Lower local_lower;
            fast_zero(local_lower);

            PerPixel_Upper local_upper;
            fast_zero(local_upper);

            if (is_valid) {
                packed_half2x2 color_transmittance;
                fast_copy(color_transmittance, bucket_color_transmittance_scaled[i]);
                color_transmittance.xy = __hmul2_rn(color_transmittance.xy, TINYGS_UNSCALE_HALF2);
                color_transmittance.zw = __hmul2_rn(color_transmittance.zw, TINYGS_UNSCALE_HALF2);

                local_upper.grad_color_pixel_rg = make_half2(
                    grad_image[physical_pixel_idx],
                    grad_image[physical_pixel_idx + channel_stride]
                );
                local_upper.grad_color_pixel_b_last_contributor = make_half2(
                    grad_image[physical_pixel_idx + channel_stride * 2],
                    __ushort_as_half(tile_n_contributions[physical_pixel_idx])
                );
                local_lower.xy = __hsub2_rn(make_half2(
                    /*r*/ image[physical_pixel_idx],
                    /*g*/ image[physical_pixel_idx + channel_stride]), color_transmittance.xy);
                local_lower.zw = make_half2(
                    /*b*/ __hsub(image[physical_pixel_idx + channel_stride * 2], color_transmittance.zw.x),
                    /*t*/ color_transmittance.zw.y);
            }
            // local_lower.color_pixel_after = local_lower.color_pixel_after - make_float3(color_transmittance);
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
            REG.as_uint4 = warp.shfl_up(REG.as_uint4, 1);

            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;
            const bool valid_general = valid_primitive && valid_pixel && idx < config::block_size_blend;
            const float2 off = make_float2(dx, dy) + 0.5f;
            const float2 delta = (mean2d - off / 16.0f) * 16.0f;
            // const float3 delta_coefs = make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);

            const __half2 delta_h = __float22half2_rn(delta / 16.0f);
            // const __half delta_coefs_h_xx = __hmul(delta_h.x, delta_h.x);
            // const __half delta_coefs_h_xy = __hmul(delta_h.x, delta_h.y);
            // const __half delta_coefs_h_yy = __hmul(delta_h.y, delta_h.y);
            const __half conic_x = __float2half_rn(conic.x);
            const __half conic_z = __float2half_rn(conic.z);
            const __half conic_y = __float2half_rn(conic.y);
            const __half conic_x_dx = __hmul(__hmul(conic_x, delta_h.x), h16); // conic.x * delta.x
            const __half conic_z_dy = __hmul(__hmul(conic_z, delta_h.y), h16); // conic.z * delta.y
            const __half conic_y_dy = __hmul(__hmul(conic_y, delta_h.y), h16); // conic.y * delta.y
            const __half conic_x_dxx = __hmul(delta_h.x, conic_x_dx);
            const __half conic_z_dyy = __hmul(delta_h.y, conic_z_dy);
            const __half conic_y_dxy = __hmul(delta_h.x, conic_y_dy);
            const __half quad = __hadd(conic_x_dxx, conic_z_dyy);
            const __half sigma_over_2_h = __hmul(__hfma(h0_5, quad, conic_y_dxy), h16);
            const float gaussian = __expf(-fmaxf(__half2float(sigma_over_2_h), 0.f));

            //! We have to compute another bf16 version to guarantee the non-vanishing gradient
            const __nv_bfloat162 delta_bf16 = __float22bfloat162_rn(delta / 16.0f);
            const __nv_bfloat16 delta_coefs_bf16_xx = __hmul(delta_bf16.x, delta_bf16.x);
            const __nv_bfloat16 delta_coefs_bf16_xy = __hmul(delta_bf16.x, delta_bf16.y);
            const __nv_bfloat16 delta_coefs_bf16_yy = __hmul(delta_bf16.y, delta_bf16.y);
            const __nv_bfloat16 conic_x_bf16 = __float2bfloat16_rn(conic.x);
            const __nv_bfloat16 conic_z_bf16 = __float2bfloat16_rn(conic.z);
            const __nv_bfloat16 conic_y_bf16 = __float2bfloat16_rn(conic.y);
            const __nv_bfloat16 conic_x_dx_bf16 = __hmul(__hmul(conic_x_bf16, delta_bf16.x), __float2bfloat16(16.0f)); // conic.x * delta.x
            const __nv_bfloat16 conic_z_dy_bf16 = __hmul(__hmul(conic_z_bf16, delta_bf16.y), __float2bfloat16(16.0f)); // conic.z * delta.y
            const __nv_bfloat16 conic_y_dx_bf16 = __hmul(__hmul(conic_y_bf16, delta_bf16.x), __float2bfloat16(16.0f)); // conic.y * delta.x
            const __nv_bfloat16 conic_y_dy_bf16 = __hmul(__hmul(conic_y_bf16, delta_bf16.y), __float2bfloat16(16.0f)); // conic.y * delta.y


            // leader thread loads values from shared memory into registers
            if (lane_idx == 0 && valid_general) {
                float4* dst_view = reinterpret_cast<float4*>(&REG);
                // asm this. fuck
                asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                    : "=f"(dst_view->x), "=f"(dst_view->y)
                    : "l"(saddr_upper + (i % 32) * sizeof(PerPixel_Upper)));
                asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                    : "=f"(dst_view->z), "=f"(dst_view->w)
                    : "l"(saddr_lower + (i % 32) * sizeof(PerPixel_Lower)));
            }
            __syncwarp(); // Synchronize after reading from shared memory

            const bool skip = !valid_general || tile_primitive_idx >= REG.full.grad_color_pixel_b_last_contributor.y;
            const float alpha_prepare = opacity * gaussian;
            const float color_dot_grad_color_pixel = __half2float(dot3(
              color, reinterpret_cast<const packed_half2x2&>(REG)));
            float alpha = 0.f;
            if (!skip) [[likely]] {
                alpha = fminf(alpha_prepare, config::max_fragment_alpha);
            }

            const float transmittance = __half2float(REG.full.color_pixel_after_b_transmittance.y);
            const float blending_weight = transmittance * alpha;
            const float one_minus_alpha = 1.0f - alpha;
            // color gradient
            // const float3 dL_dcolor = blending_weight * grad_color_pixel;
            const float3 dL_dcolor = blending_weight * make_float3(
              __half2float(REG.full.grad_color_pixel_rg.x),
              __half2float(REG.full.grad_color_pixel_rg.y),
              __half2float(__ushort_as_half(REG.full.grad_color_pixel_b_last_contributor.x))
            );
            dL_dcolor_accum += dL_dcolor;
            // color_pixel_after -= blending_weight * color;
            REG.full.color_pixel_after_rg = __hfma2(
              make_half2(__float2half(-blending_weight), __float2half(-blending_weight)),
              color.xy,
              REG.full.color_pixel_after_rg
            );
            REG.full.color_pixel_after_b_transmittance.x = __hfma(
              __float2half(-blending_weight),
              color.zw.x,
              REG.full.color_pixel_after_b_transmittance.x
            );

            // const float color_pixel_after_dot_grad_color_pixel = dot(color_pixel_after, grad_color_pixel);
            const float color_pixel_after_dot_grad_color_pixel = __half2float(
                dot3(reinterpret_cast<const packed_half2x2 &>(REG.parts.upper),
                     REG.parts.lower));

            //! Here is the problem of half, the gradient of conic.x * delta.x is too small if we are using half.
            const float2 prepare_dl_dmean2d = __bfloat1622float2({
                __hadd(conic_x_dx_bf16, conic_y_dy_bf16),
                __hadd(conic_y_dx_bf16, conic_z_dy_bf16)
            });

            // alpha gradient
            const float dL_dalpha_from_color = transmittance * color_dot_grad_color_pixel - color_pixel_after_dot_grad_color_pixel / one_minus_alpha;
            const float dL_draw_opacity_partial = alpha * dL_dalpha_from_color;
            // dL_draw_opacity_partial_accum += dL_draw_opacity_partial;
            dL_draw_opacity_partial_accum += dL_draw_opacity_partial;

            // conic and mean2d gradient
            const float3 dL_dconic = -0.5f * dL_draw_opacity_partial * make_float3(
                __bfloat162float(delta_coefs_bf16_xx),
                __bfloat162float(delta_coefs_bf16_xy),
                __bfloat162float(delta_coefs_bf16_yy)
            ) * 256;
            // dL_dconic_accum += dL_dconic;
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = dL_draw_opacity_partial * prepare_dl_dmean2d;

            // dL_dmean2d_accum -= dL_dmean2d;
            dL_dmean2d_accum -= dL_dmean2d;
            absdL_dmean2d_accum += make_float2(fabsf(dL_dmean2d.x), fabsf(dL_dmean2d.y));
            // transmittance *= one_minus_alpha;
            REG.full.color_pixel_after_b_transmittance.y = __float2half_rn(transmittance * one_minus_alpha);
        }
    }

    // finally add the gradients using atomics
    if (valid_primitive) {
#ifndef NDEBUG
        // Boundary check for gradient arrays
        assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
        atomicAdd(&primitive_info_gradients[primitive_idx].mean_xy,
                  __float22half2_rn(make_float2(dL_dmean2d_accum.x, dL_dmean2d_accum.y)));
        if (absgrad_mean2d != nullptr) {
            atomicAdd(&absgrad_mean2d[primitive_idx].x, absdL_dmean2d_accum.x);
            atomicAdd(&absgrad_mean2d[primitive_idx].y, absdL_dmean2d_accum.y);
        }
        const float dL_draw_opacity = dL_draw_opacity_partial_accum * (1.0f - opacity);
        atomicAdd(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
        atomicAdd(&primitive_info_gradients[primitive_idx].conic_ab,
                  __float22half2_rn(make_float2(dL_dconic_accum.x, dL_dconic_accum.y)));
        atomicAdd(&primitive_info_gradients[primitive_idx].color_rg,
                  __float22half2_rn(make_float2(dL_dcolor_accum.x, dL_dcolor_accum.y)));
        atomicAdd(&primitive_info_gradients[primitive_idx].conic_c_color_b,
                  __float22half2_rn(make_float2(dL_dconic_accum.z, dL_dcolor_accum.z)));
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
__device__ inline void fast_zero_aligned_8b(T& val) {
    uint64_t* eight_byte = reinterpret_cast<uint64_t*>(&val);
#pragma unroll
    for (int i = 0; i < sizeof(T) / sizeof(uint64_t); ++i) {
        eight_byte[i] = 0;
    }
}

template<typename T>
__device__ inline void fast_copy_16bytes(T& dst, const T& src) {
  reinterpret_cast<uint4 &>(dst) = reinterpret_cast<const uint4 &>(src);
}

__device__ inline __half2 doth3(__half2 x1, __half2 y1, __half2 z1,
                                     __half2 x2, __half2 y2, __half2 z2) {
    return __hfma2(x1, x2, __hfma2(y1, y2, __hmul2(z1, z2)));
}


__device__ inline __nv_bfloat162 doth3(__nv_bfloat162 x1, __nv_bfloat162 y1, __nv_bfloat162 z1,
                                       __nv_bfloat162 x2, __nv_bfloat162 y2, __nv_bfloat162 z2) {
#if __CUDA_ARCH__ >= 800
    return __hfma2(x1, x2, __hfma2(y1, y2, __hmul2(z1, z2)));
#else
    return __hmul2(z1, z2) + __hmul2(x1, x2) + __hmul2(y1, y2);
#endif
}

__device__ inline float sum_float(const __half2& inc) {
    return __half2float(inc.x) + __half2float(inc.y);
}

__device__ inline float sum_float(const __nv_bfloat162& inc) {
    return __bfloat162float(inc.x) + __bfloat162float(inc.y);
}

__device__ __forceinline__ void load2a(__half2& out, const __half* mult_of_2) {
#ifdef NDEBUG
  out = reinterpret_cast<const __half2 &>(*mult_of_2);
#else
  out = make_half2(mult_of_2[0], mult_of_2[1]);
#endif
}

__device__ __forceinline__ void load2a(ushort2& out, const ushort* mult_of_2) {
#ifdef NDEBUG
  out = reinterpret_cast<const ushort2 &>(*mult_of_2);
#else
  out = make_ushort2(mult_of_2[0], mult_of_2[1]);
#endif
}

// Load from aligned global memory, 4x4Bytes
__device__ __forceinline__ void load4a_gmem(PackedPixels_Lower& dst, const packed_half2x2* src) {
    // TODO: If nvcc cannot generate proper version, consider use asm. 
    reinterpret_cast<uint4&>(dst) = *(reinterpret_cast<const uint4*>(src));
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


/* -------------------- half version -------------------- */
// 2 pixel X 1 GS per thread
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
    float2* __restrict__ absgrad_mean2d,
    // float* __restrict__ grad_conic,
    float* __restrict__ grad_raw_opacity,
    // float3* __restrict__ grad_color,
    PrimitiveInfoGradient* __restrict__ primitive_info_gradients,
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

    // corresponds to n_contributions
    ushort tile_primitive_idx;
    uint32_t tile_primitive_idx_ui32;
    if (const int tile_primitive_idx_int32 = tile_bucket_idx * 32 + lane_idx;
        tile_primitive_idx_int32 > config::max_contributions) {
      static_assert(((uint)config::max_contributions + 1u) % 32 == 0,
                    "max_contributions + 1 must be divisible by 32 (warp size).");
      return;
    } else {
      // in range => set the variable and continue.
      tile_primitive_idx = (ushort)tile_primitive_idx_int32;
      tile_primitive_idx_ui32 = (uint32_t)tile_primitive_idx | ((uint32_t)tile_primitive_idx << 16);
    }

    const int instance_idx = tile_instance_range.x + tile_primitive_idx;
    const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

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
    constexpr uint32_t one_u162 = 0x00010001u;

    // load gaussian data
    uint primitive_idx = 0;
    float2 mean2d = {0.0f, 0.0f};
    __half2 conic_x{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 conic_y{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 conic_z{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 opacity{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 color_r{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 color_g{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    __half2 color_b{CUDART_ZERO_FP16, CUDART_ZERO_FP16};
    ushort2 tile_primitive_idx2 = {0xFFFF, 0xFFFF};
    // tile metadata
    const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
    const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_width};

    bucket_color_transmittance_scaled += bucket_idx * config::block_size_blend;

    if (valid_primitive) {
        primitive_idx = instance_primitive_indices[instance_idx];
        mean2d = primitive_mean2d[primitive_idx] / 16.0f - make_float2(tile_coords);
        const PrimitiveInfo info = primitive_info[primitive_idx];
        conic_x = make_half2(__ushort_as_half(info.conic_xy.x), __ushort_as_half(info.conic_xy.x));
        conic_y = make_half2(__ushort_as_half(info.conic_xy.y), __ushort_as_half(info.conic_xy.y));
        conic_z = make_half2(__ushort_as_half(info.conic_z_raw_opacity.x), __ushort_as_half(info.conic_z_raw_opacity.x));
        const float f_opacity = activate_opacity(__half2float(__ushort_as_half(info.conic_z_raw_opacity.y)));
        opacity = make_half2(__float2half_rn(f_opacity), __float2half_rn(f_opacity));
        color_r = make_half2(__ushort2half_rn(info.rgb.x), __ushort2half_rn(info.rgb.x));
        color_g = make_half2(__ushort2half_rn(info.rgb.y), __ushort2half_rn(info.rgb.y));
        color_b = make_half2(__ushort2half_rn(info.rgb.z), __ushort2half_rn(info.rgb.z));
        tile_primitive_idx2 = {tile_primitive_idx, tile_primitive_idx};
    }

    conic_x = __hmul2(conic_x, h_16_2);
    conic_y = __hmul2(conic_y, h_16_2);
    conic_z = __hmul2(conic_z, h_16_2);

    color_r = __hmul2(color_r, TINYGS_UNSCALE_HALF2);
    color_g = __hmul2(color_g, TINYGS_UNSCALE_HALF2);
    color_b = __hmul2(color_b, TINYGS_UNSCALE_HALF2);



    //? Gradient accumulation, kept in float, we are operating one GS's gradients
    //? we do not need half since most half precision operations are about pixels
    float2 dL_dmean2d_accum = {0.0f, 0.0f};
    float2 absdL_dmean2d_accum = {0.0f, 0.0f};
    float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
    float dL_draw_opacity_partial_accum = 0.0f;
    float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

    alignas(16) PackedPixels_Upper REGup;   fast_zero_aligned_8b(REGup);
    alignas(16) PackedPixels_Lower REGlow;  fast_zero_aligned_8b(REGlow);

    constexpr int warp_size = 32;
    constexpr int warp_size_2 = warp_size * 2;
    // One warp is capable of processing 64 pixels at a time in this half version.
    __shared__ PackedPixels_Upper cached_per_pixel_all_upper[config::blend_bwd_n_warps][warp_size];
    __shared__ PackedPixels_Lower cached_per_pixel_all_lower[config::blend_bwd_n_warps][warp_size];
    auto& cached_per_pixel_lower = cached_per_pixel_all_lower[warp_idx];
    auto& cached_per_pixel_upper = cached_per_pixel_all_upper[warp_idx];
    const uint lane_idx_uint = static_cast<uint>(lane_idx); // thread_idx in the warp, 0 <= lane_idx_uint < 32

    // iterate over all pixels in the tile
    constexpr int total_pixel_padded = config::tile_width * config::tile_width + warp_size_2 - 1;
    for (uint ii = 0; ii < total_pixel_padded; ii += warp_size_2) {
        // --- fetch data if not the tail ---
        if (ii < config::block_size_blend) {
            const uint width_in_tile = (width + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
            const uint height_in_tile = (height + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
            const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
            // 0 <= i < 256, since ii < total_pixel_padded < 256 and ii % 64 == 0
            const uint i = ii + lane_idx_uint * 2;
            const uint local_tile = i >> (2 * tinygs::kImageTileLog2);                              // 0..3
            const uint intile = i % (tinygs::kImageTile * tinygs::kImageTile);                      // 0..63
            const uint dx = (intile % tinygs::kImageTile) + (local_tile % 2) * tinygs::kImageTile;  // 0..16
            const uint dy = (intile / tinygs::kImageTile) + (local_tile / 2) * tinygs::kImageTile;  // 0..16
            const uint2 pixel_coords = {start_pixel_coords.x + dx, start_pixel_coords.y + dy};
            const uint physical_pixel_idx = tinygs::get_linear_index_tiled(
                /* row */ pixel_coords.y, /* col */ pixel_coords.x, width_in_tile);
            const bool is_valid = pixel_coords.x < width && pixel_coords.y < height &&
                dx < config::tile_width && dy < config::tile_width;

            PackedPixels_Lower local_lower;
            fast_zero_aligned_8b(local_lower);

            PackedPixels_Upper local_upper;
            fast_zero_aligned_8b(local_upper);

            // Assumes i valid indicates i+1 valid, this is true if the width % 2 == 0, 
            // which is always the case in our application.
            if (is_valid) {
                // 1. Load Global Memory
                load4a_gmem(local_lower, bucket_color_transmittance_scaled + i);

                __half2 image_color_r; load2a(image_color_r, image + physical_pixel_idx);
                __half2 image_color_g; load2a(image_color_g, image + physical_pixel_idx + channel_stride);
                __half2 image_color_b; load2a(image_color_b, image + physical_pixel_idx + channel_stride * 2);

                // Compose the results
                load2a(local_upper.grad_color_r, grad_image + physical_pixel_idx);
                load2a(local_upper.grad_color_g, grad_image + physical_pixel_idx + channel_stride);
                load2a(local_upper.grad_color_b, grad_image + physical_pixel_idx + channel_stride * 2);
                load2a(local_upper.last_contributor, tile_n_contributions + physical_pixel_idx);
                local_lower.color_after_r = __hfma2(local_lower.color_after_r, __hneg2(TINYGS_UNSCALE_HALF2), image_color_r);
                local_lower.color_after_g = __hfma2(local_lower.color_after_g, __hneg2(TINYGS_UNSCALE_HALF2), image_color_g);
                local_lower.color_after_b = __hfma2(local_lower.color_after_b, __hneg2(TINYGS_UNSCALE_HALF2), image_color_b);
                local_lower.transmittance = __hmul2(local_lower.transmittance, TINYGS_UNSCALE_HALF2);
            }
            // Store to shared
            cached_per_pixel_upper[lane_idx] = local_upper;
            cached_per_pixel_lower[lane_idx] = local_lower;
            __syncwarp(); // Synchronize after writing to shared memory
        }

        // --- do actural computation ---
        // although the upper bound of j is 32, but deal with 2 pixel per thread/iteration.
        for (uint j = 0; j < warp_size; ++j) {
            const uint i = ii + j * 2;
            // which pixel index should this thread deal with?
            // overflow is ok, will much greater than the block size, and mark invalid
            const uint idx = i - 2 * lane_idx_uint;
            const uint local_tile = idx >> (2 * tinygs::kImageTileLog2); // 0..3
            const uint intile = idx % (tinygs::kImageTile * tinygs::kImageTile); // 0..63
            const uint dx = (intile % tinygs::kImageTile) + (local_tile % 2) * tinygs::kImageTile; // 0..16
            const uint dy = (intile / tinygs::kImageTile) + (local_tile / 2) * tinygs::kImageTile; // 0..16
            // This is the 0- pixel's position.
            const uint2 pixel_coords = {start_pixel_coords.x + dx, start_pixel_coords.y + dy};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;
            const bool valid_general = valid_primitive && valid_pixel && idx < config::block_size_blend;

            // This pixel information
            const float2 off = make_float2(dx, dy) + 0.5f;
            const float2 off_div_16 = off / 16.0f;
            const float2 delta0_f = mean2d - off_div_16;
            const float2 delta1_f = mean2d - (off + make_float2(1.0f, 0.0f)) / 16.0f;


            const __half2 delta_x = make_half2(__float2half_rn(delta0_f.x), __float2half_rn(delta1_f.x));
            const __half2 delta_y = make_half2(__float2half_rn(delta0_f.y), __float2half_rn(delta1_f.y));

            const __half2 conic_x_dx = __hmul2(conic_x, delta_x); // conic.x * delta.x
            const __half2 conic_y_dx = __hmul2(conic_y, delta_x); // conic.y * delta.x
            const __half2 conic_z_dy = __hmul2(conic_z, delta_y); // conic.z * delta.y
            const __half2 conic_y_dy = __hmul2(conic_y, delta_y); // conic.y * delta.y
            const __half2 conic_x_dxx = __hmul2(delta_x, conic_x_dx);
            const __half2 conic_z_dyy = __hmul2(delta_y, conic_z_dy);
            const __half2 conic_y_dxy = __hmul2(delta_x, conic_y_dy);
            const __half2 quad = __hadd2(conic_x_dxx, conic_z_dyy);
            const __half2 sigma_over_2_h = __hmul2(__hfma2_relu(h0_5_2, quad, conic_y_dxy), h_16_2);
            const __half2 gaussian = h2exp(__hneg2(sigma_over_2_h));

            const __half2 dxdx = __hmul2(delta_x, delta_x);
            const __half2 dxdy = __hmul2(delta_x, delta_y);
            const __half2 dydy = __hmul2(delta_y, delta_y);


            { // Prepare the upper part of the register
              uint4 &regup = reinterpret_cast<uint4 &>(REGup);
              regup = warp.shfl_up(regup, 1);
              if (lane_idx == 0)
                fast_copy_16bytes(REGup, cached_per_pixel_upper[j]);
            }

            { // Prepare the lower part of the register
              uint4 &reglow = reinterpret_cast<uint4 &>(REGlow);
              reglow = warp.shfl_up(reglow, 1);
              if (lane_idx == 0)
                fast_copy_16bytes(REGlow, cached_per_pixel_lower[j]);
            }

            // const bool skip = !valid_general || tile_primitive_idx >= REG.parts.grad_color_pixel_b_last_contributor.y;
            const uint enable_mask = (valid_general ? 0xFFFFFFFFu : 0u) &
                                      __vcmpltu2(tile_primitive_idx_ui32, REGup.last_contributor_ui32);

            __half2 alpha_prepare = __hmul2(opacity, gaussian);
            // alpha is set to zero if not enabled.
            reinterpret_cast<uint32_t&>(alpha_prepare) &= enable_mask;
            // const float color_dot_grad_color_pixel = __half2float(dot3(color, reinterpret_cast<const packed_half2x2&>(REG)));
            const __half2 color_dot_grad_color_pixel = doth3(
                color_r, color_g, color_b,
                REGup.grad_color_r, REGup.grad_color_g, REGup.grad_color_b
            );
            const __half2 alpha = __hmin2(alpha_prepare, h_max_fragment_alpha_2);

            //! small alpha should be skipped
            uint32_t enable = __hle2_mask(alpha, make_half2(CUDART_MIN_DENORM_FP16, CUDART_MIN_DENORM_FP16));

            // we have set the maximum transmittance to be about 0.99, and alpha is always larger than half precision.
            const __half2 transmittance = REGlow.transmittance;
            const __half2 blending_weight = __hmul2(transmittance, alpha);
            const __half2 one_minus_alpha = __hsub2(h_1_2, alpha);

            // --- color gradient ---
            // const float3 dL_dcolor = blending_weight * grad_color_pixel;
            const __half2 dl_dcolor_r = __hmul2(blending_weight, REGup.grad_color_r);
            dL_dcolor_accum.x += sum_float(dl_dcolor_r);
            const __half2 dl_dcolor_g = __hmul2(blending_weight, REGup.grad_color_g);
            dL_dcolor_accum.y += sum_float(dl_dcolor_g);
            const __half2 dl_dcolor_b = __hmul2(blending_weight, REGup.grad_color_b);
            dL_dcolor_accum.z += sum_float(dl_dcolor_b);

            // --- update reg ---
            // color_pixel_after -= blending_weight * color;
            REGlow.color_after_r = __hfma2(__hneg2(blending_weight), color_r, REGlow.color_after_r);
            REGlow.color_after_g = __hfma2(__hneg2(blending_weight), color_g, REGlow.color_after_g);
            REGlow.color_after_b = __hfma2(__hneg2(blending_weight), color_b, REGlow.color_after_b);

            const __half2 color_pixel_after_dot_grad_color_pixel = doth3(
                REGlow.color_after_r, REGlow.color_after_g, REGlow.color_after_b,
                REGup.grad_color_r, REGup.grad_color_g, REGup.grad_color_b
            );

            //! Here is the problem of half, the gradient of conic.x * delta.x is too small if we are using half.
            const __half2 prepare_dl_dmean2d_x = __hadd2(conic_x_dx, conic_y_dy);
            const __half2 prepare_dl_dmean2d_y = __hadd2(conic_y_dx, conic_z_dy);

            // alpha gradient
            // const float dL_dalpha_from_color = transmittance * color_dot_grad_color_pixel - color_pixel_after_dot_grad_color_pixel / one_minus_alpha;
            // const float dL_draw_opacity_partial = alpha * dL_dalpha_from_color;
            const __half2 dL_dalpha_from_color = __hfma2(transmittance, color_dot_grad_color_pixel,
                                                         __hneg2(__h2div(color_pixel_after_dot_grad_color_pixel, one_minus_alpha)));
            const __half2 dL_draw_opacity_partial = __hmul2(alpha, dL_dalpha_from_color);

            // dL_draw_opacity_partial_accum += dL_draw_opacity_partial;
            dL_draw_opacity_partial_accum += sum_float(dL_draw_opacity_partial);
            // conic and mean2d gradient
            const __half2 dL_draw_opacity_partial_neg128 =
                __hmul2(dL_draw_opacity_partial, __float22half2_rn(make_float2(-128.f, -128.f)));

            __half2 dL_dconic_x = __hmul2(dL_draw_opacity_partial_neg128, dxdx);
            reinterpret_cast<uint32_t&>(dL_dconic_x) &= enable_mask;
            __half2 dL_dconic_y = __hmul2(dL_draw_opacity_partial_neg128, dxdy);
            reinterpret_cast<uint32_t&>(dL_dconic_y) &= enable_mask;
            __half2 dL_dconic_z = __hmul2(dL_draw_opacity_partial_neg128, dydy);
            reinterpret_cast<uint32_t&>(dL_dconic_z) &= enable_mask;

            // dL_dconic_accum += dL_dconic;
            dL_dconic_accum.x += sum_float(dL_dconic_x);
            dL_dconic_accum.y += sum_float(dL_dconic_y);
            dL_dconic_accum.z += sum_float(dL_dconic_z);

            // const float2 dL_dmean2d = dL_draw_opacity_partial * prepare_dl_dmean2d;
            __half2 dL_dmean2d_x = __hmul2(dL_draw_opacity_partial, prepare_dl_dmean2d_x);
            reinterpret_cast<uint32_t&>(dL_dmean2d_x) &= enable_mask;
            __half2 dL_dmean2d_y = __hmul2(dL_draw_opacity_partial, prepare_dl_dmean2d_y);
            reinterpret_cast<uint32_t&>(dL_dmean2d_y) &= enable_mask;

            // dL_dmean2d_accum -= dL_dmean2d;
            const float2 dL_dmean2d = {sum_float(dL_dmean2d_x), sum_float(dL_dmean2d_y)};
            dL_dmean2d_accum -= dL_dmean2d;
            absdL_dmean2d_accum += make_float2(fabsf(dL_dmean2d.x), fabsf(dL_dmean2d.y));
            // transmittance *= one_minus_alpha;
            REGlow.transmittance = __hmul2(REGlow.transmittance, one_minus_alpha);
        }
    }

    // finally add the gradients using atomics
    if (valid_primitive) {
#ifndef NDEBUG
        // Boundary check for gradient arrays
        assert(primitive_idx >= 0 && primitive_idx < n_primitives);
#endif
        atomicAdd(&primitive_info_gradients[primitive_idx].mean_xy,
                  __float22half2_rn(make_float2(dL_dmean2d_accum.x, dL_dmean2d_accum.y)));
        if (absgrad_mean2d != nullptr) {
            atomicAdd(&absgrad_mean2d[primitive_idx].x, absdL_dmean2d_accum.x);
            atomicAdd(&absgrad_mean2d[primitive_idx].y, absdL_dmean2d_accum.y);
        }
        const float dL_draw_opacity = dL_draw_opacity_partial_accum * (1.0f - __half2float(opacity.x));
        atomicAdd(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
        atomicAdd(&primitive_info_gradients[primitive_idx].conic_ab,
                  __float22half2_rn(make_float2(dL_dconic_accum.x, dL_dconic_accum.y)));
        atomicAdd(&primitive_info_gradients[primitive_idx].color_rg,
                  __float22half2_rn(make_float2(dL_dcolor_accum.x, dL_dcolor_accum.y)));
        atomicAdd(&primitive_info_gradients[primitive_idx].conic_c_color_b,
                  __float22half2_rn(make_float2(dL_dconic_accum.z, dL_dcolor_accum.z)));
    }
}


} // namespace tinygs::fast_gs_fp16::kernels::backward
