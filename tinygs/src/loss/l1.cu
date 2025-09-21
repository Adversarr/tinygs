#include "tinygs/cuda/common_device.cuh"
#include "tinygs/loss/l1.hpp"
#include <cuda_fp16.h>
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

/**
 * Computes L1 loss with clamped predictions. Equivalent to L1Loss in PyTorch.
 *
 * Predictions are clamped to [0,1] range before computing loss.
 * L1 = |clamp(pred, 0, 1) - target|_1 . mean()
 *
 * The gradient is computed as:
 *    grad = sign(clamp(pred, 0, 1) - target) / N
 */

// Optimized float specialization
__global__ void l1_kernel(int N, const float *__restrict__ pred,
                                 const float *__restrict__ target,
                                 float *__restrict__ loss,
                                 float *__restrict__ grad, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const float p_raw = pred[i];
  const float p = saturate(p_raw);  // Clamp p to [0,1]
  const float t = target[i];
  const float diff = p - t;
  const float l = fabsf(diff);  // Use fabsf for float instead of fabs
  const float g = copysignf(1.0f, diff);  // More efficient than conditional
  const float g_sat = g * saturate_deriv(p_raw);  // Derivative of saturate
  
  if (loss != nullptr) {
    loss[i] = fmaf(l, scale, loss[i]);  // Use fused multiply-add
  }
  if (grad != nullptr) {
    grad[i] = fmaf(g, scale, grad[i]);  // Use fused multiply-add
  }
}


void L1Loss::evaluate(LossContext ctx) {
  int n = ctx.pred.size();
  NVTX3_FUNC_RANGE();

  const float actual_scale = ctx.scale / n;
  linear_kernel(l1_kernel, 0, ctx.stream, n,
    static_cast<const float*>(ctx.pred.data),
    static_cast<const float*>(ctx.target.data),
    static_cast<float*>(ctx.loss.data),
    static_cast<float*>(ctx.grad.data),
    actual_scale);
  tinygs::maybe_sync(ctx.stream);
}

} // namespace tinygs