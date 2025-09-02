#include "tinygs/cuda/common_device.cuh"
#include "tinygs/loss/l1.hpp"
#include <cuda_fp16.h>

template <typename T>
__global__ void l1_kernel(        //
    int N,                        //
    const T *__restrict__ pred,   //
    const T *__restrict__ target, //
    T *__restrict__ loss,         //
    T *__restrict__ grad,         //
    float scale);

// Optimized float specialization
template <>
__global__ void l1_kernel<float>(int N, const float *__restrict__ pred,
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
  template <typename T> void L1Loss<T>::evaluate(LossContext<T> ctx) {
  int n = ctx.pred.size();
  linear_kernel(l1_kernel<T>, 0, ctx.stream, n, ctx.pred.data, ctx.target.data,
                ctx.loss.data, ctx.grad.data, ctx.scale);
}

template class L1Loss<float>;

} // namespace tinygs