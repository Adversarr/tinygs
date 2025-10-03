#include "tinygs/cuda/common_device.cuh"
#include "tinygs/loss/l2.hpp"
#include <cuda_fp16.h>
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

// L2 loss kernel with clamped predictions
__global__ void l2_kernel(int N, const float *__restrict__ pred,
                          const float *__restrict__ target,
                          float *__restrict__ loss,
                          float *__restrict__ grad, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const float p_raw = pred[i];
  const float p = saturate(p_raw);
  const float t = target[i];
  const float diff = p - t;
  const float l = 0.5f * diff * diff;
  const float g = diff;

  if (loss != nullptr) {
    loss[i] = fmaf(l, scale, loss[i]);
  }
  if (grad != nullptr) {
    grad[i] = fmaf(g, scale, grad[i]);
  }
}

void L2Loss::evaluate(LossContext ctx, float scale) {
  int n = ctx.pred.shape.padded_size();
  int npix = ctx.pred.shape.size();
  NVTX3_FUNC_RANGE();

  const float actual_scale = scale / npix;
  linear_kernel(l2_kernel, 0, ctx.stream, n,
                static_cast<const float *>(ctx.pred.data),
                static_cast<const float *>(ctx.target.data),
                static_cast<float *>(ctx.loss.data),
                static_cast<float *>(ctx.grad.data), actual_scale);
  tinygs::maybe_sync(ctx.stream);
}

} // namespace tinygs