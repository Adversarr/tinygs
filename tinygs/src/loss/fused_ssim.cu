#include "tinygs/cuda/common_device.cuh"
#include "tinygs/cuda/gpu_memory.hpp"
#include <cooperative_groups.h>
#include <tinygs/loss/fused_ssim.hpp>
#include <memory>
#include <nvtx3/nvtx3.hpp>

#include <cuda_fp16.hpp>

#include "tinygs/cuda/vec.hpp"

namespace cg = cooperative_groups;

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

using namespace tinygs;

// ------------------------------------------
// Forward Kernel: Fused SSIM
//  - Two-pass convolution to get mu1, mu2,
//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes partial derivatives
//    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
// ------------------------------------------
__global__ void fused_ssim_cuda_fp32(
    int H,
    int W,
    float C1,
    float C2,
    float scale,
    const float* __restrict__ img1, // pred
    const float* __restrict__ img2, // target
    float* __restrict__ ssim_map, // loss per pixel
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12
) {
    constexpr float cGauss[11] = {
        0.001028380123898387f,
        0.0075987582094967365f,
        0.036000773310661316f,
        0.10936068743467331f,
        0.21300552785396576f,
        0.26601171493530273f,
        0.21300552785396576f,
        0.10936068743467331f,
        0.036000773310661316f,
        0.0075987582094967365f,
        0.001028380123898387f
    };


    auto block = cg::this_thread_block();
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const uint width_in_tile = (W + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint height_in_tile = (H + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    const uint physical_pixel_idx = get_linear_index_tiled(pix_y, pix_x, width_in_tile);

    // const int pix_id = pix_y * W + pix_x;
    // const int num_pix = H * W;

    // Shared memory for the tile (img1, img2)
    // __shared__ float sTile[SHARED_Y][SHARED_X][2];
    // __shared__ __half2 sTile[SHARED_Y][SHARED_X];
    __shared__ __half2 sTile[SHARED_Y][SHARED_X];
    // After horizontal pass, store partial sums here
    // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
    __shared__ float xconv[CONV_Y][CONV_X][5];

    // Each block processes B x C sub-batches. We loop over channels:
    for (int c = 0; c < 3; ++c) {
        // ------------------------------------------------------------
        // 1) Load (img1, img2) tile + halo into shared memory
        // ------------------------------------------------------------
        {
            const int this_warp = block.thread_rank() / 32;         // 0..7
            const int lane_id = block.thread_rank() % 32;           // 0..31
            const int warp_dx = this_warp / 2;                      // 0..3
            const int warp_dy = this_warp % 2;                      // 0, 1
            // alternative, we loop over a fixed grid: 2row, 4col, 4x4*2 tile(two tile per row) => 4x4 * 4x4 = 32x32 load
            const int tile_start_x = (block.group_index().x * BLOCK_X + (warp_dx - 1) * kImageTile);
            const int tile_start_y = (block.group_index().y * BLOCK_Y + (warp_dy == 0 ? -1 : 1) * (kImageTile));
            #pragma unroll 4
            for (int local_linear_idx = lane_id;                 // 0..32
                 local_linear_idx < (2 * kImageTile * kImageTile); // 128
                 local_linear_idx += 32) {
                const int local_x = local_linear_idx % kImageTile; // 0..8
                const int local_y = local_linear_idx / kImageTile; // 0..16
                assert(local_x >= 0 && local_x < 8);
                assert(local_y >= 0 && local_y < 16);
                const int gx = tile_start_x + local_x;
                const int gy = tile_start_y + local_y;
                const int stile_x = (warp_dx - 1) * kImageTile + local_x + 5; // -3...30
                const int stile_y = (warp_dy == 0 ? -1 : 1) * (kImageTile) + local_y + 5; // -3...30
                assert(-3 <= stile_x && stile_x < 30);
                assert(-3 <= stile_y && stile_y < 30);
                const auto physical_pixel_idx = get_linear_index_tiled(gy, gx, width_in_tile);

                __half2 xy(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
                if (gx < W && gy < H && 0 <= gx && 0 <= gy) [[likely]] {
                    xy = __float22half2_rn({img1[physical_pixel_idx + c * channel_stride],
                                            img2[physical_pixel_idx + c * channel_stride]});
                }
                if (stile_x >= 0 && stile_x < SHARED_X && stile_y >= 0 && stile_y < SHARED_Y) {
                    sTile[stile_y][stile_x] = xy;
                }
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 2) Horizontal convolution (11x1) in shared memory
        //    We'll accumulate symmetrical pairs around center.
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y;
            // int lx = threadIdx.x + HALO;  // skip left halo

            float sumX   = 0.f;
            float sumX2  = 0.f;
            float sumY   = 0.f;
            float sumY2  = 0.f;
            float sumXY  = 0.f;

#pragma unroll
            for (int d = 0; d < HALO * 2 + 1; ++d) {
                const float w = cGauss[d];
                const __half2 stile_item = sTile[ly][threadIdx.x + d];
                const float X = __half2float(stile_item.x);
                const float Y = __half2float(stile_item.y);
                sumX  += X * w;
                sumX2 += (X * X) * w;
                sumY  += Y * w;
                sumY2 += (Y * Y) * w;
                sumXY += (X * Y) * w;
            }

            // Write out partial sums
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            // Possibly handle second row in same warp
            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX   = 0.f; sumX2  = 0.f;
                sumY   = 0.f; sumY2  = 0.f;
                sumXY  = 0.f;

                for (int d = 0; d < HALO * 2 + 1; ++d) {
                    const float w = cGauss[d];
                    const __half2 stile_item = sTile[ly2][threadIdx.x + d];
                    const float X = __half2float(stile_item.x);
                    const float Y = __half2float(stile_item.y);
                    sumX  += X * w;
                    sumX2 += (X * X) * w;
                    sumY  += Y * w;
                    sumY2 += (Y * Y) * w;
                    sumXY += (X * Y) * w;
                }

                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 3) Vertical convolution (1x11) + final SSIM
        // ------------------------------------------------------------
        {
            // int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
            for (int d = 0; d < HALO * 2 + 1; ++d) {
              const float w = cGauss[d];
              float *current = xconv[threadIdx.y + d][lx];
              out0 += current[0] * w;
              out1 += current[1] * w;
              out2 += current[2] * w;
              out3 += current[3] * w;
              out4 += current[4] * w;
            }

            if (pix_x < W && pix_y < H) {
#ifdef TINYGS_SSIM_ENABLE_BOUNDARY
                // Skip boundary pixels - incomplete 11x11 Gaussian window
                if (pix_x < HALO || pix_x >= W - HALO || 
                    pix_y < HALO || pix_y >= H - HALO) {
                    continue;
                }
#endif
                const uint global_idx = c * channel_stride + physical_pixel_idx;

                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12   = out4 - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);
                ssim_map[global_idx] += (1 - val) * scale;

                if (dm_dmu1) {
                    float d_m_dmu1 = (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    float d_m_dsigma12   = (2.f * C_) / (A * B);

                    dm_dmu1[global_idx]       = d_m_dmu1;
                    dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                    dm_dsigma12[global_idx]   = d_m_dsigma12;
                }
            }
        }
    }
}

// ------------------------------------------
// Backward Kernel: Apply chain rule to get
//    dL/d(img1) from partial derivatives
//    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
//    and dL/dmap (the gradient from above).
// ------------------------------------------
__global__ void fused_ssim_fp32_backward(
    int H,
    int W,
    float scale,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ dL_dimg1, // out: dL/dpred
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    constexpr float cGauss[11] = {
        0.001028380123898387f,
        0.0075987582094967365f,
        0.036000773310661316f,
        0.10936068743467331f,
        0.21300552785396576f,
        0.26601171493530273f,
        0.21300552785396576f,
        0.10936068743467331f,
        0.036000773310661316f,
        0.0075987582094967365f,
        0.001028380123898387f
    };

    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    // const int pix_id = pix_y * W + pix_x;
    // const int num_pix = H * W;

    const uint width_in_tile = (W + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint height_in_tile = (H + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    const uint physical_pixel_idx = get_linear_index_tiled(pix_y, pix_x, width_in_tile);

    const float neg_scale = -scale;
    // Shared memory for the fused data:
    // [0]: dm_dmu1*dL, [1]: dm_dsigma1_sq*dL, [2]: dm_dsigma12*dL
    __shared__ float sData[3][SHARED_Y][SHARED_X];
    __shared__ float sScratch[CONV_Y][CONV_X][3];

    for (int c = 0; c < 3; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = img1[channel_stride * c + physical_pixel_idx];
            p2 = img2[channel_stride * c + physical_pixel_idx];
        }

        // (1) Load + fuse multiplication
        {
            const int this_warp = block.thread_rank() / 32;         // 0..7
            const int lane_id = block.thread_rank() % 32;           // 0..31
            const int warp_dx = this_warp / 2;                      // 0..3
            const int warp_dy = this_warp % 2;                      // 0, 1
            const int tile_start_x = (block.group_index().x * BLOCK_X + (warp_dx - 1) * kImageTile);
            const int tile_start_y = (block.group_index().y * BLOCK_Y + (warp_dy == 0 ? -1 : 1) * (kImageTile));
            #pragma unroll 4
            for (int local_linear_idx = lane_id;                 // 0..32
                 local_linear_idx < (2 * kImageTile * kImageTile); // 128
                 local_linear_idx += 32) {
                const int local_x = local_linear_idx % kImageTile; // 0..8
                const int local_y = local_linear_idx / kImageTile; // 0..16
                assert(local_x >= 0 && local_x < 8);
                assert(local_y >= 0 && local_y < 16);
                const int gx = tile_start_x + local_x;
                const int gy = tile_start_y + local_y;
                const int stile_x = (warp_dx - 1) * kImageTile + local_x + 5; // -3...30
                const int stile_y = (warp_dy == 0 ? -1 : 1) * (kImageTile) + local_y + 5; // -3...30
                assert(-3 <= stile_x && stile_x < 30);
                assert(-3 <= stile_y && stile_y < 30);
                const auto physical_pixel_idx_gygx = get_linear_index_tiled(gy, gx, width_in_tile);

                float vmu  = 0.f;
                float vs1  = 0.f;
                float vs12  = 0.f;
                if (gx < W && gy < H && 0 <= gx && 0 <= gy) [[likely]] {
                    vmu  = dm_dmu1[channel_stride * c + physical_pixel_idx_gygx];
                    vs1  = dm_dsigma1_sq[channel_stride * c + physical_pixel_idx_gygx];
                    vs12 = dm_dsigma12[channel_stride * c + physical_pixel_idx_gygx];
                }
                if (stile_x >= 0 && stile_x < SHARED_X && stile_y >= 0 && stile_y < SHARED_Y) {
                    sData[0][stile_y][stile_x] = vmu  ;
                    sData[1][stile_y][stile_x] = vs1  ;
                    sData[2][stile_y][stile_x] = vs12 ;
                }
            }
        }
        block.sync();

        // (2) Horizontal pass
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y;
                if (yy < CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float left0  = sData[0][yy][lx - d];
                        float left1  = sData[1][yy][lx - d];
                        float left2  = sData[2][yy][lx - d];

                        float right0 = sData[0][yy][lx + d];
                        float right1 = sData[1][yy][lx + d];
                        float right2 = sData[2][yy][lx + d];

                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    // center
                    {
                        float wc = cGauss[HALO];
                        float c0 = sData[0][yy][lx];
                        float c1 = sData[1][yy][lx];
                        float c2 = sData[2][yy][lx];
                        accum0 += c0 * wc;
                        accum1 += c1 * wc;
                        accum2 += c2 * wc;
                    }

                    sScratch[yy][threadIdx.x][0] = accum0;
                    sScratch[yy][threadIdx.x][1] = accum1;
                    sScratch[yy][threadIdx.x][2] = accum2;
                }
            }
        }
        block.sync();

        // (3) Vertical pass -> finalize dL/d(img1)
        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = sScratch[ly - d][lx];
                float* bot = sScratch[ly + d][lx];

                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            // center
            {
                float wc = cGauss[HALO];
                float* ctr = sScratch[ly][lx];
                sum0 += ctr[0] * wc;
                sum1 += ctr[1] * wc;
                sum2 += ctr[2] * wc;
            }

            // final accumulation
            float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

            dL_dimg1[channel_stride * c + physical_pixel_idx] += neg_scale * dL_dpix; // NOTE: (1 - ssim)
        }
        block.sync();
    }
}

// fp16 forward kernel
__global__ void fused_ssim_cuda_fp16(
    int H,
    int W,
    float C1,
    float C2,
    float scale,
    const __half* __restrict__ img1,
    const __half* __restrict__ img2,
    __half* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    constexpr float cGauss[11] = {
        0.001028380123898387f,
        0.0075987582094967365f,
        0.036000773310661316f,
        0.10936068743467331f,
        0.21300552785396576f,
        0.26601171493530273f,
        0.21300552785396576f,
        0.10936068743467331f,
        0.036000773310661316f,
        0.0075987582094967365f,
        0.001028380123898387f
    };

    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const uint width_in_tile = (W + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint height_in_tile = (H + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    const uint physical_pixel_idx = get_linear_index_tiled(pix_y, pix_x, width_in_tile);

    __shared__ __half2 sTile[SHARED_Y][SHARED_X];
    __shared__ float xconv[CONV_Y][CONV_X][5];

    for (int c = 0; c < 3; ++c) {
        // load tile
        {
            const int this_warp = block.thread_rank() / 32;
            const int lane_id = block.thread_rank() % 32;
            const int warp_dx = this_warp / 2;
            const int warp_dy = this_warp % 2;
            const int tile_start_x = (block.group_index().x * BLOCK_X + (warp_dx - 1) * kImageTile);
            const int tile_start_y = (block.group_index().y * BLOCK_Y + (warp_dy == 0 ? -1 : 1) * (kImageTile));
            #pragma unroll 4
            for (int local_linear_idx = lane_id;
                 local_linear_idx < (2 * kImageTile * kImageTile);
                 local_linear_idx += 32) {
                const int local_x = local_linear_idx % kImageTile;
                const int local_y = local_linear_idx / kImageTile;
                const int gx = tile_start_x + local_x;
                const int gy = tile_start_y + local_y;
                const int stile_x = (warp_dx - 1) * kImageTile + local_x + 5;
                const int stile_y = (warp_dy == 0 ? -1 : 1) * (kImageTile) + local_y + 5;
                const auto physical_pixel_idx_gygx = get_linear_index_tiled(gy, gx, width_in_tile);

                __half2 xy(CUDART_ZERO_FP16, CUDART_ZERO_FP16);
                if (gx < W && gy < H && 0 <= gx && 0 <= gy) [[likely]] {
                    __half h1 = img1[physical_pixel_idx_gygx + c * channel_stride];
                    __half h2 = img2[physical_pixel_idx_gygx + c * channel_stride];
                    xy = __halves2half2(h1, h2);
                }
                if (stile_x >= 0 && stile_x < SHARED_X && stile_y >= 0 && stile_y < SHARED_Y) {
                    sTile[stile_y][stile_x] = xy;
                }
            }
        }
        block.sync();

        // horizontal pass
        {
            int ly = threadIdx.y;
            float sumX   = 0.f;
            float sumX2  = 0.f;
            float sumY   = 0.f;
            float sumY2  = 0.f;
            float sumXY  = 0.f;

#pragma unroll
            for (int d = 0; d < HALO * 2 + 1; ++d) {
                const float w = cGauss[d];
                const __half2 stile_item = sTile[ly][threadIdx.x + d];
                const float X = __half2float(stile_item.x);
                const float Y = __half2float(stile_item.y);
                sumX  += X * w;
                sumX2 += (X * X) * w;
                sumY  += Y * w;
                sumY2 += (Y * Y) * w;
                sumXY += (X * Y) * w;
            }
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX   = 0.f; sumX2  = 0.f;
                sumY   = 0.f; sumY2  = 0.f;
                sumXY  = 0.f;
                for (int d = 0; d < HALO * 2 + 1; ++d) {
                    const float w = cGauss[d];
                    const __half2 stile_item = sTile[ly2][threadIdx.x + d];
                    const float X = __half2float(stile_item.x);
                    const float Y = __half2float(stile_item.y);
                    sumX  += X * w;
                    sumX2 += (X * X) * w;
                    sumY  += Y * w;
                    sumY2 += (Y * Y) * w;
                    sumXY += (X * Y) * w;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // vertical pass + final
        {
            int lx = threadIdx.x;
            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;
#pragma unroll
            for (int d = 0; d < HALO * 2 + 1; ++d) {
                const float w = cGauss[d];
                float *current = xconv[threadIdx.y + d][lx];
                out0 += current[0] * w;
                out1 += current[1] * w;
                out2 += current[2] * w;
                out3 += current[3] * w;
                out4 += current[4] * w;
            }
            if (pix_x < W && pix_y < H) {
#ifdef TINYGS_SSIM_ENABLE_BOUNDARY
                // Skip boundary pixels - incomplete 11x11 Gaussian window
                if (pix_x < HALO || pix_x >= W - HALO || 
                    pix_y < HALO || pix_y >= H - HALO) {
                    continue;
                }
#endif
                const uint global_idx = c * channel_stride + physical_pixel_idx;

                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12   = out4 - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;
                float val = (C_ * D_) / (A * B);

                if (ssim_map) {
                    float cur = __half2float(ssim_map[global_idx]);
                    ssim_map[global_idx] = __float2half_rn(fmaf((1.f - val), scale, cur));
                }
                if (dm_dmu1) {
                    float d_m_dmu1 = (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    float d_m_dsigma12   = (2.f * C_) / (A * B);
                    dm_dmu1[global_idx]       = d_m_dmu1;
                    dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                    dm_dsigma12[global_idx]   = d_m_dsigma12;
                }
            }
        }
    }
}

// fp16 backward kernel
__global__ void fused_ssim_fp16_backward(
    int H,
    int W,
    float scale,
    const __half* __restrict__ img1,
    const __half* __restrict__ img2,
    __half* __restrict__ dL_dimg1,
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    constexpr float cGauss[11] = {
        0.001028380123898387f,
        0.0075987582094967365f,
        0.036000773310661316f,
        0.10936068743467331f,
        0.21300552785396576f,
        0.26601171493530273f,
        0.21300552785396576f,
        0.10936068743467331f,
        0.036000773310661316f,
        0.0075987582094967365f,
        0.001028380123898387f
    };

    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;

    const uint width_in_tile = (W + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint height_in_tile = (H + tinygs::kImageTileMask) >> tinygs::kImageTileLog2;
    const uint channel_stride = width_in_tile * height_in_tile << (2 * tinygs::kImageTileLog2);
    const uint physical_pixel_idx = get_linear_index_tiled(pix_y, pix_x, width_in_tile);

    const float neg_scale = -scale;
    __shared__ float sData[3][SHARED_Y][SHARED_X];
    __shared__ float sScratch[CONV_Y][CONV_X][3];

    for (int c = 0; c < 3; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = __half2float(img1[channel_stride * c + physical_pixel_idx]);
            p2 = __half2float(img2[channel_stride * c + physical_pixel_idx]);
        }
        // load derivatives tile
        {
            const int this_warp = block.thread_rank() / 32;
            const int lane_id = block.thread_rank() % 32;
            const int warp_dx = this_warp / 2;
            const int warp_dy = this_warp % 2;
            const int tile_start_x = (block.group_index().x * BLOCK_X + (warp_dx - 1) * kImageTile);
            const int tile_start_y = (block.group_index().y * BLOCK_Y + (warp_dy == 0 ? -1 : 1) * (kImageTile));
            #pragma unroll 4
            for (int local_linear_idx = lane_id;
                 local_linear_idx < (2 * kImageTile * kImageTile);
                 local_linear_idx += 32) {
                const int local_x = local_linear_idx % kImageTile;
                const int local_y = local_linear_idx / kImageTile;
                const int gx = tile_start_x + local_x;
                const int gy = tile_start_y + local_y;
                const int stile_x = (warp_dx - 1) * kImageTile + local_x + 5;
                const int stile_y = (warp_dy == 0 ? -1 : 1) * (kImageTile) + local_y + 5;
                const auto physical_pixel_idx_gygx = get_linear_index_tiled(gy, gx, width_in_tile);

                float vmu = 0.f, vs1 = 0.f, vs12 = 0.f;
                if (gx < W && gy < H && 0 <= gx && 0 <= gy) [[likely]] {
                    vmu  = dm_dmu1[channel_stride * c + physical_pixel_idx_gygx];
                    vs1  = dm_dsigma1_sq[channel_stride * c + physical_pixel_idx_gygx];
                    vs12 = dm_dsigma12[channel_stride * c + physical_pixel_idx_gygx];
                }
                if (stile_x >= 0 && stile_x < SHARED_X && stile_y >= 0 && stile_y < SHARED_Y) {
                    sData[0][stile_y][stile_x] = vmu;
                    sData[1][stile_y][stile_x] = vs1;
                    sData[2][stile_y][stile_x] = vs12;
                }
            }
        }
        block.sync();

        // horizontal
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;
            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * BLOCK_Y;
                if (yy < CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;
#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float left0  = sData[0][yy][lx - d];
                        float left1  = sData[1][yy][lx - d];
                        float left2  = sData[2][yy][lx - d];
                        float right0 = sData[0][yy][lx + d];
                        float right1 = sData[1][yy][lx + d];
                        float right2 = sData[2][yy][lx + d];
                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    float wc = cGauss[HALO];
                    float c0 = sData[0][yy][lx];
                    float c1 = sData[1][yy][lx];
                    float c2 = sData[2][yy][lx];
                    accum0 += c0 * wc;
                    accum1 += c1 * wc;
                    accum2 += c2 * wc;
                    sScratch[yy][threadIdx.x][0] = accum0;
                    sScratch[yy][threadIdx.x][1] = accum1;
                    sScratch[yy][threadIdx.x][2] = accum2;
                }
            }
        }
        block.sync();

        // vertical + finalize
        if (pix_x < W && pix_y < H) {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = sScratch[ly - d][lx];
                float* bot = sScratch[ly + d][lx];
                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            float wc = cGauss[HALO];
            float* ctr = sScratch[ly][lx];
            sum0 += ctr[0] * wc;
            sum1 += ctr[1] * wc;
            sum2 += ctr[2] * wc;
            float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;
            const uint out_idx = channel_stride * c + physical_pixel_idx;
            float cur = __half2float(dL_dimg1[out_idx]);
            dL_dimg1[out_idx] = __float2half_rn(fmaf(neg_scale, dL_dpix, cur));
        }
        block.sync();
    }
}

namespace tinygs {

struct FusedSSIMLoss::Impl {
  GPUBuffer<float> dm_dmu1;
  GPUBuffer<float> dm_dsigma1_sq;
  GPUBuffer<float> dm_dsigma12;

  void ensure(size_t total, cudaStream_t stream) {
    if (!dm_dmu1 || dm_dmu1.size() < total) {
      dm_dmu1 = GPUBuffer<float>(stream, total);
    }
    if (!dm_dsigma1_sq || dm_dsigma1_sq.size() < total) {
      dm_dsigma1_sq = GPUBuffer<float>(stream, total);
    }
    if (!dm_dsigma12 || dm_dsigma12.size() < total) {
      dm_dsigma12 = GPUBuffer<float>(stream, total);
    }
  }
};

FusedSSIMLoss::~FusedSSIMLoss() = default;

FusedSSIMLoss::FusedSSIMLoss() {
  m_impl = std::make_unique<Impl>();
}

struct m_domain { static constexpr char const* name{"fused_ssim"}; };
struct m_fused_ssim_fwd { static constexpr char const* message{"forward"}; };
struct m_fused_ssim_bwd { static constexpr char const* message{"backward"}; };
using regstr = nvtx3::registered_string_in<m_domain>;
using range  = nvtx3::scoped_range_in<m_domain>;

void FusedSSIMLoss::evaluate(LossContext ctx, float scale) {
    NVTX3_FUNC_RANGE();
    const auto data_type = ctx.target.data_type;
    if (data_type != ctx.pred.data_type) {
      throw std::runtime_error("FusedSSIMLoss: prediction and target data type must be the same");
    }

    int H = ctx.pred.shape.height;
    int W = ctx.pred.shape.width;
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y,
              /*batch_size*/ 1);
    dim3 block(BLOCK_X, BLOCK_Y);
    int total = ctx.pred.shape.padded_size();   // physical
    m_impl->ensure(total, ctx.stream);
    const float actual_scale = scale / ctx.pred.shape.size(); // normalize by element count

    if (data_type == DataType::Float32) {
      const float* pred = static_cast<const float*>(ctx.pred.data);
      const float* targ = static_cast<const float*>(ctx.target.data);
      float* loss = static_cast<float*>(ctx.loss.data);
      float* grad = static_cast<float*>(ctx.grad.data);

      if (ctx.grad) {
        {
          auto msg = regstr::get<m_fused_ssim_fwd>();
          nvtx3::event_attributes attr(msg, nvtx3::payload{total});
          range range(attr);

          fused_ssim_cuda_fp32<<<grid, block, 0, ctx.stream>>>(
              H, W, m_c1, m_c2, actual_scale,
              pred,
              targ,
              loss,
              m_impl->dm_dmu1.data(),
              m_impl->dm_dsigma1_sq.data(),
              m_impl->dm_dsigma12.data());
          tinygs::maybe_sync(ctx.stream);
        }
        {
          auto msg = regstr::get<m_fused_ssim_bwd>();
          nvtx3::event_attributes attr(msg, nvtx3::payload{total});
          range range(attr);

          fused_ssim_fp32_backward<<<grid, block, 0, ctx.stream>>>(
              H, W,
              actual_scale,
              pred,
              targ,
              grad,
              m_impl->dm_dmu1.data(),
              m_impl->dm_dsigma1_sq.data(),
              m_impl->dm_dsigma12.data());
          tinygs::maybe_sync(ctx.stream);
        }
      } else {
        auto msg = regstr::get<m_fused_ssim_fwd>();
        nvtx3::event_attributes attr(msg, nvtx3::payload{total});
        range range(attr);
        fused_ssim_cuda_fp32<<<grid, block, 0, ctx.stream>>>(
            H, W, m_c1, m_c2, actual_scale,
            pred,
            targ,
            loss,
            nullptr, nullptr, nullptr);
        tinygs::maybe_sync(ctx.stream);
      }
    } else if (data_type == DataType::Float16) {
      const __half* pred = static_cast<const __half*>(ctx.pred.data);
      const __half* targ = static_cast<const __half*>(ctx.target.data);
      __half* loss = static_cast<__half*>(ctx.loss.data);
      __half* grad = static_cast<__half*>(ctx.grad.data);

      if (ctx.grad) {
        {
          auto msg = regstr::get<m_fused_ssim_fwd>();
          nvtx3::event_attributes attr(msg, nvtx3::payload{total});
          range range(attr);

          fused_ssim_cuda_fp16<<<grid, block, 0, ctx.stream>>>(
              H, W, m_c1, m_c2, actual_scale,
              pred,
              targ,
              loss,
              m_impl->dm_dmu1.data(),
              m_impl->dm_dsigma1_sq.data(),
              m_impl->dm_dsigma12.data());
          tinygs::maybe_sync(ctx.stream);
        }
        {
          auto msg = regstr::get<m_fused_ssim_bwd>();
          nvtx3::event_attributes attr(msg, nvtx3::payload{total});
          range range(attr);

          fused_ssim_fp16_backward<<<grid, block, 0, ctx.stream>>>(
              H, W,
              actual_scale,
              pred,
              targ,
              grad,
              m_impl->dm_dmu1.data(),
              m_impl->dm_dsigma1_sq.data(),
              m_impl->dm_dsigma12.data());
          tinygs::maybe_sync(ctx.stream);
        }
      } else {
        auto msg = regstr::get<m_fused_ssim_fwd>();
        nvtx3::event_attributes attr(msg, nvtx3::payload{total});
        range range(attr);
        fused_ssim_cuda_fp16<<<grid, block, 0, ctx.stream>>>(
            H, W, m_c1, m_c2, actual_scale,
            pred,
            targ,
            loss,
            nullptr, nullptr, nullptr);
        tinygs::maybe_sync(ctx.stream);
      }
    } else {
      throw std::runtime_error("FusedSSIMLoss: unsupported data type");
    }
}

} // namespace tinygs