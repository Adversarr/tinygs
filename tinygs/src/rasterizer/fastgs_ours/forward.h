/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../../helper_math.h"
#include <functional>
#include <tuple>
#include "tinygs/core/gaussian.hpp"
namespace fast_gs::rasterization {

    std::tuple<int, int, int, int, int> forward(
        std::function<char*(size_t)> per_primitive_buffers_func,
        std::function<char*(size_t)> per_tile_buffers_func,
        std::function<char*(size_t)> per_instance_buffers_func,
        std::function<char*(size_t)> per_bucket_buffers_func,
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* opacities_raw,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        tinygs::DensificationInfo* densification_info,
        float* image,
        float* alpha,
        const int n_primitives,
        const int active_sh_bases,
        const int total_bases_sh_rest,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near,
        const float far,
        cudaStream_t major_stream,
        cudaStream_t helper_stream,
        char* zero_copy,
        cudaEvent_t memset_per_tile_done,
        cudaEvent_t copy_n_instances_done,
        cudaEvent_t preprocess_done);

}
