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
__global__ void l1_kernel_f32(int N, const float *__restrict__ pred,
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

// fp16 scalar specialization
__global__ void l1_kernel_f16(int N, const half *__restrict__ pred,
                              const half *__restrict__ target,
                              half *__restrict__ loss,
                              half *__restrict__ grad, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const float p_raw = __half2float(pred[i]);
  const float p = saturate(p_raw);
  const float t = __half2float(target[i]);
  const float diff = p - t;
  const float l = fabsf(diff);
  const float g = copysignf(1.0f, diff);

  if (loss != nullptr) {
    const float cur = __half2float(loss[i]);
    loss[i] = __float2half_rn(fmaf(l, scale, cur));
  }
  if (grad != nullptr) {
    const float cur = __half2float(grad[i]);
    grad[i] = __float2half_rn(fmaf(g, scale, cur));
  }
}

// fp16 half2 SIMD specialization (processes 2 elements per thread)
__global__ void l1_kernel_f16_h2(int N_pairs, const __half2 *__restrict__ pred,
                                 const __half2 *__restrict__ target,
                                 __half2 *__restrict__ loss,
                                 __half2 *__restrict__ grad, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_pairs) {
    return;
  }

  const float2 p_raw = __half22float2(pred[i]);
  const float2 t = __half22float2(target[i]);

  const float p0 = saturate(p_raw.x);
  const float p1 = saturate(p_raw.y);
  const float diff0 = p0 - t.x;
  const float diff1 = p1 - t.y;

  const float l0 = fabsf(diff0);
  const float l1 = fabsf(diff1);
  const float g0 = copysignf(1.0f, diff0);
  const float g1 = copysignf(1.0f, diff1);

  if (loss != nullptr) {
    float2 cur = __half22float2(loss[i]);
    cur.x = fmaf(l0, scale, cur.x);
    cur.y = fmaf(l1, scale, cur.y);
    loss[i] = __floats2half2_rn(cur.x, cur.y);
  }
  if (grad != nullptr) {
    float2 cur = __half22float2(grad[i]);
    cur.x = fmaf(g0, scale, cur.x);
    cur.y = fmaf(g1, scale, cur.y);
    grad[i] = __floats2half2_rn(cur.x, cur.y);
  }
}


void L1Loss::evaluate(LossContext ctx, float scale) {
  const auto data_type = ctx.target.data_type;
  if (ctx.pred.data_type != data_type) {
    throw std::runtime_error(fmt::format(
        "L1Loss: prediction and target data type must be the same, got {} vs. {}",
        to_string(ctx.pred.data_type), to_string(data_type)));
  }

  int n = ctx.pred.shape.padded_size(); // physical
  int npix = ctx.pred.shape.size();     // actual
  NVTX3_FUNC_RANGE();

  const float actual_scale = scale / npix;

  if (data_type == DataType::Float32) {
    linear_kernel(l1_kernel_f32, 0, ctx.stream, n,
      static_cast<const float*>(ctx.pred.data),
      static_cast<const float*>(ctx.target.data),
      static_cast<float*>(ctx.loss.data),
      static_cast<float*>(ctx.grad.data),
      actual_scale);
  } else if (data_type == DataType::Float16) {
    if ((n & 1) == 0) {
      // Use half2 SIMD when the element count is divisible by 2
      linear_kernel(l1_kernel_f16_h2, 0, ctx.stream, n / 2,
        reinterpret_cast<const __half2*>(ctx.pred.data),
        reinterpret_cast<const __half2*>(ctx.target.data),
        reinterpret_cast<__half2*>(ctx.loss.data),
        reinterpret_cast<__half2*>(ctx.grad.data),
        actual_scale);
    } else {
      linear_kernel(l1_kernel_f16, 0, ctx.stream, n,
        static_cast<const half*>(ctx.pred.data),
        static_cast<const half*>(ctx.target.data),
        static_cast<half*>(ctx.loss.data),
        static_cast<half*>(ctx.grad.data),
        actual_scale);
    }
  } else {
    throw std::runtime_error("L1Loss: only float32 is supported");
  }
  tinygs::maybe_sync(ctx.stream);
}

} // namespace tinygs