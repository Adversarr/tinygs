#include "tinygs/cuda/common_device.cuh"
#include "tinygs/loss/l1.hpp"
#include <cuda_fp16.h>

/**
 * Computes L1 loss. Equivalent to L1Loss in PyTorch.
 *
 * L1 = |pred - target|_1 . mean()
 *
 * The gradient is computed as:
 *    grad = sign(pred - target) / N
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

  const float p = pred[i];
  const float t = target[i];
  const float diff = p - t;
  const float l = fabsf(diff);  // Use fabsf for float instead of fabs
  const float g = copysignf(1.0f, diff);  // More efficient than conditional
  
  if (loss != nullptr) {
    loss[i] = fmaf(l, scale, loss[i]);  // Use fused multiply-add
  }
  if (grad != nullptr) {
    grad[i] = fmaf(g, scale, grad[i]);  // Use fused multiply-add
  }
}

namespace tinygs {

void L1Loss::evaluate(LossContext ctx) {
  int n = ctx.pred.size();
  const float actual_scale = ctx.scale / n;
  linear_kernel(l1_kernel, 0, ctx.stream, n,
    static_cast<const float*>(ctx.pred.data),
    static_cast<const float*>(ctx.target.data),
    static_cast<float*>(ctx.loss.data),
    static_cast<float*>(ctx.grad.data),
    actual_scale);
}

} // namespace tinygs