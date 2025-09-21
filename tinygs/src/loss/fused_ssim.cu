#include "cuda/gpu_memory.hpp"
#include <cooperative_groups.h>
#include <tinygs/loss/fused_ssim.hpp>
#include <memory>
#include <nvtx3/nvtx3.hpp>

#include "tinygs/cuda/vec.hpp"

namespace cg = cooperative_groups;

// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
__constant__ float cGauss[11] = {
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
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
__device__ __forceinline__ vec3 get_pix_value2(
    const vec3* img,
    int y, int x,
    int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    // HWC layout: [B, H, W, C]
    return img[y * W + x];
}

// ------------------------------------------
// Forward Kernel: Fused SSIM
//  - Two-pass convolution to get mu1, mu2,
//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes partial derivatives
//    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
// ------------------------------------------
__global__ void fusedssimCUDA2(
    int H,
    int W,
    float C1,
    float C2,
    float scale,
    const vec3* __restrict__ img1, // pred
    const vec3* __restrict__ img2, // target
    vec3* __restrict__ ssim_map, // loss per pixel
    vec3* __restrict__ dm_dmu1,
    vec3* __restrict__ dm_dsigma1_sq,
    vec3* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;  // batch index
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // Shared memory for the tile (img1, img2)
    __shared__ vec3 sTile[SHARED_Y][SHARED_X][2]; // pad 2 pixel in x-axis.
    // After horizontal pass, store partial sums here
    // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
    __shared__ vec3 xconv[CONV_Y][CONV_X][5];

    // ------------------------------------------------------------
    // 1) Load (img1, img2) tile + halo into shared memory
    // ------------------------------------------------------------
    {
        const float* img1_flt = (const float*)img1;
        const float* img2_flt = (const float*)img2;

        constexpr int tileSize = SHARED_Y * SHARED_X; // total pixels in tile.
        constexpr int threads = BLOCK_X * BLOCK_Y;    // launch threads per block.
        
        const int tileStartY = block.group_index().y * BLOCK_Y;
        const int tileStartX = block.group_index().x * BLOCK_X;

        // now load is pixel by pixel.
        constexpr int steps = (tileSize + threads - 1) / threads; // how much step should I use to load everything.
        vec3 xs[steps], ys[steps]; // each thread load steps.
#pragma unroll
        for (int s = 0; s < steps; ++s) {
            int tid = s * threads + block.thread_rank();
            const int local_y = tid / SHARED_X;
            const int local_x = tid % SHARED_X;
            const int gy = tileStartY + local_y - HALO;
            const int gx = tileStartX + local_x - HALO;
            const bool valid = 0 <= gx && gx < W && 0 <= gy && gy < H;
            const vec3 x = get_pix_value2(img1, gy, gx, H, W);
            const vec3 y = get_pix_value2(img2, gy, gx, H, W);
            xs[s] = x;
            ys[s] = y;
        }
        // write to shared memory.
        #pragma unroll
        for (int s = 0; s < steps; ++s) {
            int tid = s * threads + block.thread_rank();
            if (tid < tileSize) {
                const int local_y = tid / SHARED_X;
                const int local_x = tid % SHARED_X;
                sTile[local_y][local_x][0] = xs[s];
                sTile[local_y][local_x][1] = ys[s];
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
        int lx = threadIdx.x + HALO;  // skip left halo

        vec3 sumX {0.f};
        vec3 sumX2{0.f};
        vec3 sumY {0.f};
        vec3 sumY2{0.f};
        vec3 sumXY{0.f};

        // #pragma unroll for those 5 pairs
#pragma unroll
        for (int d = 1; d <= HALO; ++d) {
            float w = cGauss[HALO - d];
            const vec3 Xleft  = sTile[ly][lx - d][0];
            const vec3 Yleft  = sTile[ly][lx - d][1];
            const vec3 Xright = sTile[ly][lx + d][0];
            const vec3 Yright = sTile[ly][lx + d][1];

            sumX  += (Xleft + Xright) * w;
            sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
            sumY  += (Yleft + Yright) * w;
            sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
            sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
        }
        // center
        {
            const vec3 centerX = sTile[ly][lx][0];
            const vec3 centerY = sTile[ly][lx][1];
            float wc = cGauss[HALO];
            sumX  += centerX * wc;
            sumX2 += (centerX * centerX) * wc;
            sumY  += centerY * wc;
            sumY2 += (centerY * centerY) * wc;
            sumXY += (centerX * centerY) * wc;
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
            sumX   = vec3(0.f); sumX2  = vec3(0.f);
            sumY   = vec3(0.f); sumY2  = vec3(0.f);
            sumXY  = vec3(0.f);

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                const vec3 Xleft  = sTile[ly2][lx - d][0];
                const vec3 Yleft  = sTile[ly2][lx - d][1];
                const vec3 Xright = sTile[ly2][lx + d][0];
                const vec3 Yright = sTile[ly2][lx + d][1];

                sumX  += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY  += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            // center
            {
                const vec3 cx = sTile[ly2][lx][0];
                const vec3 cy = sTile[ly2][lx][1];
                float wc = cGauss[HALO];
                sumX  += cx * wc;
                sumX2 += (cx * cx) * wc;
                sumY  += cy * wc;
                sumY2 += (cy * cy) * wc;
                sumXY += (cx * cy) * wc;
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
        int ly = threadIdx.y + HALO;
        int lx = threadIdx.x;

        vec3 out0 = vec3(0.f), out1 = vec3(0.f), out2 = vec3(0.f), out3 = vec3(0.f), out4 = vec3(0.f);

#pragma unroll
        for (int d = 1; d <= HALO; ++d) {
            float w = cGauss[HALO - d];
            const vec3* top = xconv[ly - d][lx];
            const vec3* bot = xconv[ly + d][lx];

            out0 += (top[0] + bot[0]) * w;
            out1 += (top[1] + bot[1]) * w;
            out2 += (top[2] + bot[2]) * w;
            out3 += (top[3] + bot[3]) * w;
            out4 += (top[4] + bot[4]) * w;
        }
        // center
        {
            float wC = cGauss[HALO];
            const vec3* ctr = xconv[ly][lx];
            out0 += ctr[0] * wC;
            out1 += ctr[1] * wC;
            out2 += ctr[2] * wC;
            out3 += ctr[3] * wC;
            out4 += ctr[4] * wC;
        }

        if (pix_x < W && pix_y < H) {
            vec3 mu1 = out0;
            vec3 mu2 = out2;
            vec3 mu1_sq = mu1 * mu1;
            vec3 mu2_sq = mu2 * mu2;

            vec3 sigma1_sq = out1 - mu1_sq;
            vec3 sigma2_sq = out3 - mu2_sq;
            vec3 sigma12   = out4 - mu1 * mu2;

            vec3 A = mu1_sq + mu2_sq + C1;
            vec3 B = sigma1_sq + sigma2_sq + C2;
            vec3 C_ = 2.f * mu1 * mu2 + C1;
            vec3 D_ = 2.f * sigma12 + C2;

            vec3 val = (C_ * D_) / (A * B);

            // int global_idx = ((bIdx * num_pix + pix_id) * CH) + c; // HWC indexing
            int global_idx = (pix_y * W + pix_x);
            ssim_map[global_idx] += (1.0f - val) * scale; // NOTE: 1 - ssim is loss

            if (dm_dmu1) {
                // partial derivatives
                const vec3 d_m_dmu1 = (
                    (mu2 * 2.f * D_) / (A * B)
                    - (mu2 * 2.f * C_) / (A * B)
                    - (mu1 * 2.f * C_ * D_) / (A * A * B)
                    + (mu1 * 2.f * C_ * D_) / (A * B * B)
                );
                vec3 d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                vec3 d_m_dsigma12   = (2.f * C_) / (A * B);

                dm_dmu1[global_idx]       = d_m_dmu1;
                dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                dm_dsigma12[global_idx]   = d_m_dsigma12;
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
__global__ void fusedssim_backwardCUDA2(
    int H,
    int W,
    float scale,
    const vec3* __restrict__ img1,
    const vec3* __restrict__ img2,
    vec3* __restrict__ dL_dimg1, // out: dL/dpred
    const vec3* __restrict__ dm_dmu1,
    const vec3* __restrict__ dm_dsigma1_sq,
    const vec3* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();

    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;
    const int bIdx   = block.group_index().z;

    // Shared memory for the fused data:
    // [0]: dm_dmu1*dL, [1]: dm_dsigma1_sq*dL, [2]: dm_dsigma12*dL
    __shared__ vec3 sData[SHARED_Y][SHARED_X][3];
    __shared__ vec3 sScratch[CONV_Y][CONV_X][3];

        vec3 p1 = vec3(0.f), p2 = vec3(0.f);
        if (pix_x < W && pix_y < H) {
            p1 = get_pix_value2(img1, pix_y, pix_x, H, W);
            p2 = get_pix_value2(img2, pix_y, pix_x, H, W);
        }

        // (1) Load + fuse multiplication
        {
            const int start_y = block.group_index().y * BLOCK_Y;
            const int start_x = block.group_index().x * BLOCK_X;

            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int warp_id = tid / 32; // 0-7
            int lane_id = tid % 32; // 0-32
            constexpr int totalThreads = BLOCK_X * BLOCK_Y;
            constexpr int num_warps = (totalThreads + 31) / 32;

            for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                if (lane_id < SHARED_X) {
                    int gy = start_y + row - HALO;
                    int gx = start_x + lane_id - HALO;
                    const float dL_dmap = (
                        gx >= HALO && gx < W - HALO && gy >= HALO && gy < H - HALO ? 
                        scale : 0.0f
                    );

                    const vec3 vmu   = get_pix_value2(dm_dmu1,       gy, gx, H, W);
                    const vec3 vs1   = get_pix_value2(dm_dsigma1_sq, gy, gx, H, W);
                    const vec3 vs12  = get_pix_value2(dm_dsigma12,   gy, gx, H, W);

                    sData[row][lane_id][0] = vmu  * dL_dmap;
                    sData[row][lane_id][1] = vs1  * dL_dmap;
                    sData[row][lane_id][2] = vs12 * dL_dmap;
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
                    vec3 accum0 = vec3(0.f), accum1 = vec3(0.f), accum2 = vec3(0.f);

#pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        vec3 left0  = sData[yy][lx - d][0];
                        vec3 left1  = sData[yy][lx - d][1];
                        vec3 left2  = sData[yy][lx - d][2];

                        vec3 right0 = sData[yy][lx + d][0];
                        vec3 right1 = sData[yy][lx + d][1];
                        vec3 right2 = sData[yy][lx + d][2];

                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    // center
                    {
                        float wc = cGauss[HALO];
                        vec3 c0 = sData[yy][lx][0];
                        vec3 c1 = sData[yy][lx][1];
                        vec3 c2 = sData[yy][lx][2];
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

            vec3 sum0 = vec3(0.f), sum1 = vec3(0.f), sum2 = vec3(0.f);

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                const vec3* const top = sScratch[ly - d][lx];
                const vec3* const bot = sScratch[ly + d][lx];

                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            // center
            {
                float wc = cGauss[HALO];
                const vec3* const ctr = sScratch[ly][lx];
                sum0 += ctr[0] * wc;
                sum1 += ctr[1] * wc;
                sum2 += ctr[2] * wc;
            }

            // final accumulation
            vec3 dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

            // int out_idx = ((bIdx * num_pix + pix_id) * CH) + c; // HWC indexing
            // dL_dimg1[out_idx] += -dL_dpix[0]; // NOTE: (1 - ssim)
            dL_dimg1[pix_id] += -1.0f * dL_dpix; // NOTE: (1 - ssim)
        }
        block.sync();
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

void FusedSSIMLoss::evaluate(LossContext ctx) {
    NVTX3_FUNC_RANGE();
    int H = ctx.pred.shape.height;
    int W = ctx.pred.shape.width;
    int CH = ctx.pred.shape.channel;
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y,
              /*batch_size*/ 1);
    dim3 block(BLOCK_X, BLOCK_Y);
    int total = H * W * CH;
    m_impl->ensure(total, ctx.stream);
    const float actual_scale = ctx.scale / (total);

    const float* pred = static_cast<float*>(ctx.pred.data);
    const float* targ = static_cast<float*>(ctx.target.data);
    float* loss = static_cast<float*>(ctx.loss.data);
    float* grad = static_cast<float*>(ctx.grad.data);

    if (ctx.grad) {
      {
        auto msg = regstr::get<m_fused_ssim_fwd>();
        nvtx3::event_attributes attr(msg, nvtx3::payload{total});
        range range(attr);

        fusedssimCUDA2<<<grid, block, 0, ctx.stream>>>(
            H, W, m_c1, m_c2, actual_scale,
            reinterpret_cast<const vec3 *>(pred),
            reinterpret_cast<const vec3 *>(targ),
            reinterpret_cast<vec3 *>(loss),
            reinterpret_cast<vec3 *>(m_impl->dm_dmu1.data()),
            reinterpret_cast<vec3 *>(m_impl->dm_dsigma1_sq.data()),
            reinterpret_cast<vec3 *>(m_impl->dm_dsigma12.data()));
        tinygs::maybe_sync(ctx.stream);
      }
      {
        auto msg = regstr::get<m_fused_ssim_bwd>();
        nvtx3::event_attributes attr(msg, nvtx3::payload{total});
        range range(attr);

        fusedssim_backwardCUDA2<<<grid, block, 0, ctx.stream>>>(
            H, W, actual_scale,
            reinterpret_cast<const vec3*>(pred),
            reinterpret_cast<const vec3*>(targ),
            reinterpret_cast<vec3*>(grad),                                   //
            reinterpret_cast<const vec3*>(m_impl->dm_dmu1.data()),
            reinterpret_cast<const vec3*>(m_impl->dm_dsigma1_sq.data()),
            reinterpret_cast<const vec3*>(m_impl->dm_dsigma12.data()));
        tinygs::maybe_sync(ctx.stream);
      }
    } else {
      auto msg = regstr::get<m_fused_ssim_fwd>();
      nvtx3::event_attributes attr(msg, nvtx3::payload{total});
      range range(attr);
      fusedssimCUDA2<<<grid, block, 0, ctx.stream>>>(
          H, W, m_c1, m_c2, actual_scale,
          reinterpret_cast<const vec3*>(pred),
          reinterpret_cast<const vec3*>(targ),
          reinterpret_cast<vec3*>(loss),
          nullptr, nullptr, nullptr);
      tinygs::maybe_sync(ctx.stream);
    }
}

} // namespace tinygs