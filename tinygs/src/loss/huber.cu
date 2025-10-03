#include "tinygs/cuda/common_device.cuh"
#include "tinygs/loss/huber.hpp"
#include <cuda_fp16.h>
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

__global__ void huber_kernel(int N, const float *__restrict__ pred,
                             const float *__restrict__ target,
                             float *__restrict__ loss,
                             float *__restrict__ grad, float scale,
                             float delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const float p_raw = pred[i];
  const float p = saturate(p_raw);
  const float t = target[i];
  const float diff = p - t;
  const float ad = fabsf(diff);

  float l, g;
  if (ad <= delta) {
    l = 0.5f * diff * diff;
    g = diff;
  } else {
    l = delta * (ad - 0.5f * delta);
    g = delta * copysignf(1.0f, diff);
  }

  if (loss != nullptr) {
    loss[i] = fmaf(l, scale, loss[i]);
  }
  if (grad != nullptr) {
    grad[i] = fmaf(g, scale, grad[i]);
  }
}

void HuberLoss::evaluate(LossContext ctx, float scale) {
  int n = ctx.pred.shape.padded_size();
  int npix = ctx.pred.shape.size();
  NVTX3_FUNC_RANGE();

  const float actual_scale = scale / npix;
  linear_kernel(huber_kernel, 0, ctx.stream, n,
                static_cast<const float *>(ctx.pred.data),
                static_cast<const float *>(ctx.target.data),
                static_cast<float *>(ctx.loss.data),
                static_cast<float *>(ctx.grad.data), actual_scale, m_delta);
  tinygs::maybe_sync(ctx.stream);
}

} // namespace tinygs