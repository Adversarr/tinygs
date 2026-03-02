/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cub/cub.cuh>
#include <functional>
#include <nvtx3/nvtx3.hpp>
#include "tinygs/cuda/common_host.hpp"

#include "buffer_utils.h"
#include "forward.h"
#include "../../helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include "utils/scope_timer.hpp"

#include "nvtx_gs.h"

std::tuple<int, int, int, int, int> fast_gs::rasterization::forward(
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
    const float near_,
    const float far_,
    cudaStream_t major_stream,
    cudaStream_t helper_stream,
    char* zero_copy,
    cudaEvent_t memset_per_tile_done,
    cudaEvent_t copy_n_instances_done,
    cudaEvent_t preprocess_done,
    bool metric_mode,
    const int* metric_map,
    int* metric_counts)
{
    using namespace gs_nvtx;
    GS_FUNC_RANGE(); // 顶层函数范围（domain=fast_gs）

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int grid_width = grid.x;



    // Per-tile buffers
    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    int& n_visible_primitives = *reinterpret_cast<int*>(zero_copy + 0 * 128); // major stream
    int& n_instances = *reinterpret_cast<int*>(zero_copy + 1 * 128); // helper stream
    int& n_buckets = *reinterpret_cast<int*>(zero_copy + 2 * 128);      // major stream

    // Per-primitive buffers
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);

    {
        GS_RANGE_SCOPE(m_memset_per_prim, C_GREEN, catM(), n_primitives);
        cudaMemsetAsync(per_primitive_buffers.n_visible_primitives, 0, sizeof(uint), major_stream);
        cudaMemsetAsync(per_primitive_buffers.n_instances, 0, sizeof(uint), major_stream);
        CHECK_CUDA(config::debug, "memset per_primitive_buffers");
        // tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_memset_per_tile, C_GREEN, catM(), n_tiles);
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, helper_stream);
        CUDA_CHECK_THROW(cudaEventRecord(memset_per_tile_done, helper_stream));
    }

    {
        GS_RANGE_SCOPE(m_preprocess, C_BLUE, catK(), n_primitives);


        kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, major_stream>>>(
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh_coefficients_0,
            sh_coefficients_rest,
            w2c,
            cam_position,
            densification_info,
            per_primitive_buffers.depth_keys.Current(),
            per_primitive_buffers.primitive_indices.Current(),
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_primitive_buffers.color,
            per_primitive_buffers.n_visible_primitives,
            per_primitive_buffers.n_instances,
            n_primitives,
            grid.x,
            grid.y,
            active_sh_bases,
            total_bases_sh_rest,
            static_cast<float>(width),
            static_cast<float>(height),
            fx,
            fy,
            cx,
            cy,
            near_,
            far_);
        CHECK_CUDA(config::debug, "preprocess");
        CUDA_CHECK_THROW(cudaEventRecord(preprocess_done, major_stream));
        tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_copy_counts_d2h, C_GRAY, catCp(), 0);
        cudaMemcpyAsync(&n_visible_primitives, per_primitive_buffers.n_visible_primitives, sizeof(uint), cudaMemcpyDeviceToHost, major_stream);
        CHECK_CUDA(config::debug, "copy n_visible_primitives D2H");
    }


    {
        GS_RANGE_SCOPE(m_copy_counts_d2h, C_GRAY, catCp(), 0);
        CUDA_CHECK_THROW(cudaStreamWaitEvent(helper_stream, preprocess_done, cudaEventWaitDefault));
        cudaMemcpyAsync(&n_instances, per_primitive_buffers.n_instances, sizeof(uint), cudaMemcpyDeviceToHost, helper_stream);
        CHECK_CUDA(config::debug, "copy n_instances D2H");
        CUDA_CHECK_THROW(cudaEventRecord(copy_n_instances_done, helper_stream));
    }

    {
        GS_RANGE_SCOPE(m_sort_depth, C_ORANGE, catC(), n_visible_primitives);
        CUDA_CHECK_THROW(cudaStreamSynchronize(major_stream));// we need to use the variable.
        cub::DeviceRadixSort::SortPairs(
            per_primitive_buffers.cub_workspace,
            per_primitive_buffers.cub_workspace_size,
            per_primitive_buffers.depth_keys,
            per_primitive_buffers.primitive_indices,
            n_visible_primitives,
            0, sizeof(int) * 8, major_stream); // TODO: major_stream
        CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Depth)");
        tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_apply_depth_order, C_BLUE, catK(), n_visible_primitives);
        kernels::forward::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives, config::block_size_apply_depth_ordering),
                                                    config::block_size_apply_depth_ordering, 0, major_stream>>>(
            per_primitive_buffers.primitive_indices.Current(),
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            n_visible_primitives);
        CHECK_CUDA(config::debug, "apply_depth_ordering");
        tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_scan_primitive, C_PURPLE, catC(), n_visible_primitives);
        cub::DeviceScan::ExclusiveSum(
            per_primitive_buffers.cub_workspace,
            per_primitive_buffers.cub_workspace_size,
            per_primitive_buffers.offset,
            per_primitive_buffers.offset,
            n_visible_primitives, major_stream);
        CHECK_CUDA(config::debug, "cub::DeviceScan::ExclusiveSum (Primitive Offsets)");
        tinygs::maybe_sync(major_stream);
    }

    // Per-instance buffers + create instances
    CUDA_CHECK_THROW(cudaStreamSynchronize(helper_stream));
    char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers>(n_instances));
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances);

    {
        GS_RANGE_SCOPE(m_create_instances, C_PINK, catK(), n_visible_primitives);
        kernels::forward::create_instances_cu<<<div_round_up(n_visible_primitives, config::block_size_create_instances),
                                                config::block_size_create_instances, 0, major_stream>>>(
            per_primitive_buffers.primitive_indices.Current(),
            per_primitive_buffers.offset,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_instance_buffers.keys.Current(),
            per_instance_buffers.primitive_indices.Current(),
            grid.x,
            n_visible_primitives);
        CHECK_CUDA(config::debug, "create_instances");
        tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_sort_tiles, C_ORANGE, catC(), n_instances);
        cub::DeviceRadixSort::SortPairs(
            per_instance_buffers.cub_workspace,
            per_instance_buffers.cub_workspace_size,
            per_instance_buffers.keys,
            per_instance_buffers.primitive_indices,
            n_instances,
            0, sizeof(ushort) * 8, major_stream); // TODO: major_stream
        CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Tile)");
        tinygs::maybe_sync(major_stream);
    }

    if (n_instances > 0) {
        GS_RANGE_SCOPE(m_extract_ranges, C_CYAN, catK(), n_instances);
        CUDA_CHECK_THROW(cudaStreamWaitEvent(major_stream, memset_per_tile_done, cudaEventWaitDefault));
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges),
                                                       config::block_size_extract_instance_ranges, 0, major_stream>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances);
        CHECK_CUDA(config::debug, "extract_instance_ranges");
        tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_bucket_counts, C_CYAN, catK(), n_tiles);
        kernels::forward::extract_bucket_counts<<<div_round_up(n_tiles, config::block_size_extract_bucket_counts),
                                                  config::block_size_extract_bucket_counts, 0, major_stream>>>(
            per_tile_buffers.instance_ranges,
            per_tile_buffers.n_buckets,
            n_tiles);
        CHECK_CUDA(config::debug, "extract_bucket_counts");
        tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_scan_buckets, C_PURPLE, catC(), n_tiles);
        cub::DeviceScan::InclusiveSum(
            per_tile_buffers.cub_workspace,
            per_tile_buffers.cub_workspace_size,
            per_tile_buffers.n_buckets,
            per_tile_buffers.bucket_offsets,
            n_tiles,
            major_stream);
        CHECK_CUDA(config::debug, "cub::DeviceScan::InclusiveSum (Bucket Counts)");
        tinygs::maybe_sync(major_stream);
    }

    {
        GS_RANGE_SCOPE(m_copy_counts_d2h, C_GRAY, catCp(), 0);
        cudaMemcpyAsync(&n_buckets, per_tile_buffers.bucket_offsets + n_tiles - 1, sizeof(uint), cudaMemcpyDeviceToHost, major_stream);
        CHECK_CUDA(config::debug, "copy n_buckets D2H");
        CUDA_CHECK_THROW(cudaStreamSynchronize(major_stream));
    }

    // Per-bucket buffers + blend
    char* per_bucket_buffers_blob = per_bucket_buffers_func(required<PerBucketBuffers>(n_buckets));
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);

    {
        GS_RANGE_SCOPE(m_blend, C_RED, catK(), n_buckets);
        if (metric_mode) {
            kernels::forward::blend_cu<true><<<grid, block, 0, major_stream>>>(
                per_tile_buffers.instance_ranges,
                per_tile_buffers.bucket_offsets,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.mean2d,
                per_primitive_buffers.conic_opacity,
                per_primitive_buffers.color,
                image,
                alpha,
                per_tile_buffers.max_n_contributions,
                per_tile_buffers.n_contributions,
                per_bucket_buffers.tile_index,
                per_bucket_buffers.color_transmittance,
                width,
                height,
                grid_width,
                n_tiles,
                metric_map,
                metric_counts);
        } else {
            kernels::forward::blend_cu<false><<<grid, block, 0, major_stream>>>(
                per_tile_buffers.instance_ranges,
                per_tile_buffers.bucket_offsets,
                per_instance_buffers.primitive_indices.Current(),
                per_primitive_buffers.mean2d,
                per_primitive_buffers.conic_opacity,
                per_primitive_buffers.color,
                image,
                alpha,
                per_tile_buffers.max_n_contributions,
                per_tile_buffers.n_contributions,
                per_bucket_buffers.tile_index,
                per_bucket_buffers.color_transmittance,
                width,
                height,
                grid_width,
                n_tiles,
                nullptr,
                nullptr);
        }
        CHECK_CUDA(config::debug, "blend");
        tinygs::maybe_sync(major_stream);
    }

    return {n_visible_primitives, n_instances, n_buckets,
            per_primitive_buffers.primitive_indices.selector,
            per_instance_buffers.primitive_indices.selector};
}