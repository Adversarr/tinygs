/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "tinygs/cuda/common_host.hpp"
#include "../../helper_math.h"
#include "kernels_backward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <functional>
#include "nvtx_gs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define WARP_SIZE 32
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float4 warp_reduce_sum_f4(float4 v) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    v.x += __shfl_xor_sync(0xffffffff, v.x, mask);
    v.y += __shfl_xor_sync(0xffffffff, v.y, mask);
    v.z += __shfl_xor_sync(0xffffffff, v.z, mask);
    v.w += __shfl_xor_sync(0xffffffff, v.w, mask);
  }
  return v;
}

/*
AoS in-place:
a: [N,16] (4x4 flattened, row-major). Results written to matrix at out_idx (16 elements).
*/
template <int NUM_THREADS = 256>
__global__ void reduce_sum_4x4_aos_inplace_f32x4_kernel(float* __restrict__ a,
                                                        int N) {
  constexpr int out_idx = 0;
  int tid = threadIdx.x;
  int lane = tid % WARP_SIZE;
  int warp = tid / WARP_SIZE;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;

  __shared__ float4 smem[NUM_WARPS][4];

  float4 s0 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 s1 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 s2 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 s3 = make_float4(0.f, 0.f, 0.f, 0.f);

  int idx = blockIdx.x * blockDim.x + tid;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride) {
    if (i == out_idx) continue; // Skip output slot
    const float* base = a + i * 16;
    float4 r0 = *reinterpret_cast<const float4*>(base +  0);
    float4 r1 = *reinterpret_cast<const float4*>(base +  4);
    float4 r2 = *reinterpret_cast<const float4*>(base +  8);
    float4 r3 = *reinterpret_cast<const float4*>(base + 12);
    s0.x += r0.x; s0.y += r0.y; s0.z += r0.z; s0.w += r0.w;
    s1.x += r1.x; s1.y += r1.y; s1.z += r1.z; s1.w += r1.w;
    s2.x += r2.x; s2.y += r2.y; s2.z += r2.z; s2.w += r2.w;
    s3.x += r3.x; s3.y += r3.y; s3.z += r3.z; s3.w += r3.w;
  }

  s0 = warp_reduce_sum_f4<WARP_SIZE>(s0);
  s1 = warp_reduce_sum_f4<WARP_SIZE>(s1);
  s2 = warp_reduce_sum_f4<WARP_SIZE>(s2);
  s3 = warp_reduce_sum_f4<WARP_SIZE>(s3);

  if (lane == 0) {
    smem[warp][0] = s0;
    smem[warp][1] = s1;
    smem[warp][2] = s2;
    smem[warp][3] = s3;
  }
  __syncthreads();

  if (warp == 0) {
    float4 t0 = (lane < NUM_WARPS) ? smem[lane][0] : make_float4(0,0,0,0);
    float4 t1 = (lane < NUM_WARPS) ? smem[lane][1] : make_float4(0,0,0,0);
    float4 t2 = (lane < NUM_WARPS) ? smem[lane][2] : make_float4(0,0,0,0);
    float4 t3 = (lane < NUM_WARPS) ? smem[lane][3] : make_float4(0,0,0,0);

    t0 = warp_reduce_sum_f4<NUM_WARPS>(t0);
    t1 = warp_reduce_sum_f4<NUM_WARPS>(t1);
    t2 = warp_reduce_sum_f4<NUM_WARPS>(t2);
    t3 = warp_reduce_sum_f4<NUM_WARPS>(t3);

    if (lane == 0) {
      float* out = a + out_idx * 16;
      atomicAdd(&out[ 0], t0.x); atomicAdd(&out[ 1], t0.y);
      atomicAdd(&out[ 2], t0.z); atomicAdd(&out[ 3], t0.w);
      atomicAdd(&out[ 4], t1.x); atomicAdd(&out[ 5], t1.y);
      atomicAdd(&out[ 6], t1.z); atomicAdd(&out[ 7], t1.w);
      atomicAdd(&out[ 8], t2.x); atomicAdd(&out[ 9], t2.y);
      atomicAdd(&out[10], t2.z); atomicAdd(&out[11], t2.w);
      atomicAdd(&out[12], t3.x); atomicAdd(&out[13], t3.y);
      atomicAdd(&out[14], t3.z); atomicAdd(&out[15], t3.w);
    }
  }
}

void fast_gs::rasterization::backward( 
    const float* grad_image,
    const float* image,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* sh1,
    const float* sh2,
    const float* sh3,
    const float4* w2c,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_bucket_buffers_blob,
    float3* grad_means,
    float3* grad_scales_raw,
    float4* grad_rotations_raw,
    float* grad_opacities_raw,
    float* grad_sh0,
    float* grad_sh1,
    float* grad_sh2,
    float* grad_sh3,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float3* grad_color,
    float4* grad_w2c,
    float4* grad_w2c_per_gs,
    tinygs::DensificationInfo* densification_info,
    float2* absgrad_mean2d_helper,
    const int n_primitives,
    const int n_visible_primitives,
    const int n_instances,
    const int n_buckets,
    const int primitive_primitive_indices_selector,
    const int instance_primitive_indices_selector,
    const int active_sh_bases,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    cudaStream_t stream
 ) {
    using namespace gs_nvtx;
    GS_FUNC_RANGE();

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;

    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances);
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);
    per_primitive_buffers.primitive_indices.selector = primitive_primitive_indices_selector;
    per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    {
        GS_RANGE_SCOPE(m_blend_backward, C_RED, catK(), n_buckets);
        const int grids = div_round_up(n_buckets, config::blend_bwd_n_warps);
        const int blocks = 32 * config::blend_bwd_n_warps;
        kernels::backward::blend_backward_cu2<<<grids, blocks, 0, stream>>>(
            per_tile_buffers.instance_ranges,
            per_tile_buffers.bucket_offsets,
            per_instance_buffers.primitive_indices.Current(),
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_primitive_buffers.color,
            grad_image,
            image,
            per_tile_buffers.max_n_contributions,
            per_tile_buffers.n_contributions,
            per_bucket_buffers.tile_index,
            per_bucket_buffers.color_transmittance,
            grad_mean2d_helper,
            absgrad_mean2d_helper,
            grad_conic_helper,
            grad_opacities_raw,
            grad_color, // used to store intermediate gradients
            n_buckets,
            n_primitives,
            width,
            height,
            grid.x);
        CHECK_CUDA(config::debug, "blend_backward");
        tinygs::maybe_sync(stream);
    }

    {
        GS_RANGE_SCOPE(m_preprocess_backward, C_BLUE, catK(), n_primitives);
        kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward),
                                                    config::block_size_preprocess_backward, 0, stream>>>(
            means,
            scales_raw,
            rotations_raw,
            sh1,
            sh2,
            sh3,
            w2c,
            cam_position,
            per_primitive_buffers.n_touched_tiles,
            grad_mean2d_helper,
            grad_conic_helper,
            absgrad_mean2d_helper,
            grad_means,
            grad_scales_raw,
            grad_rotations_raw,
            grad_color,
            grad_sh0,
            grad_sh1,
            grad_sh2,
            grad_sh3,
            grad_w2c_per_gs,
            densification_info,
            n_primitives,
            active_sh_bases,
            static_cast<float>(width),
            static_cast<float>(height),
            fx,
            fy,
            cx,
            cy);
        CHECK_CUDA(config::debug, "preprocess_backward");
        tinygs::maybe_sync(stream);
    }

    if (grad_w2c_per_gs != nullptr) {
        GS_RANGE_SCOPE(m_reduce_w2c_grad, C_GREEN, catK(), n_primitives);
        using float16 = float[16];
        const int grids = div_round_up(n_primitives, 256);
        const int blocks = 256;
        reduce_sum_4x4_aos_inplace_f32x4_kernel<<<grids, blocks, 0, stream>>>(
            reinterpret_cast<float*>(grad_w2c_per_gs),
            n_primitives);
        CUDA_CHECK_THROW(cudaMemcpyAsync(grad_w2c, grad_w2c_per_gs,
                                         16 * sizeof(float),
                                         cudaMemcpyDeviceToDevice, stream));

        CHECK_CUDA(config::debug, "reduce_sum_4x4_aos_inplace_f32x4");
        tinygs::maybe_sync(stream);
    } else {
      CUDA_CHECK_THROW(cudaMemsetAsync(grad_w2c, 0, 16 * sizeof(float), stream));
    }
}
