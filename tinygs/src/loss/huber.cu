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

// fp16 scalar specialization
__global__ void huber_kernel_f16(int N, const half *__restrict__ pred,
                                 const half *__restrict__ target,
                                 half *__restrict__ loss,
                                 half *__restrict__ grad, float scale,
                                 float delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const float p_raw = __half2float(pred[i]);
  const float p = saturate(p_raw);
  const float t = __half2float(target[i]);
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
    const float cur = __half2float(loss[i]);
    loss[i] = __float2half_rn(fmaf(l, scale, cur));
  }
  if (grad != nullptr) {
    const float cur = __half2float(grad[i]);
    grad[i] = __float2half_rn(fmaf(g, scale, cur));
  }
}

// fp16 half2 SIMD specialization
__global__ void huber_kernel_f16_h2(int N_pairs, const __half2 *__restrict__ pred,
                                    const __half2 *__restrict__ target,
                                    __half2 *__restrict__ loss,
                                    __half2 *__restrict__ grad, float scale,
                                    float delta) {
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
  const float ad0 = fabsf(diff0);
  const float ad1 = fabsf(diff1);

  float l0, g0, l1, g1;
  if (ad0 <= delta) {
    l0 = 0.5f * diff0 * diff0;
    g0 = diff0;
  } else {
    l0 = delta * (ad0 - 0.5f * delta);
    g0 = delta * copysignf(1.0f, diff0);
  }
  if (ad1 <= delta) {
    l1 = 0.5f * diff1 * diff1;
    g1 = diff1;
  } else {
    l1 = delta * (ad1 - 0.5f * delta);
    g1 = delta * copysignf(1.0f, diff1);
  }

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

void HuberLoss::evaluate(LossContext ctx, float scale) {
  const auto data_type = ctx.target.data_type;
  if (ctx.pred.data_type != data_type) {
    throw std::runtime_error("HuberLoss: prediction and target data type must be the same");
  }

  int n = ctx.pred.shape.padded_size();
  int npix = ctx.pred.shape.size();
  NVTX3_FUNC_RANGE();

  const float actual_scale = scale / npix;
  if (data_type == DataType::Float32) {
    linear_kernel(huber_kernel, 0, to_cuda_stream(ctx.queue), n,
                  static_cast<const float *>(ctx.pred.data),
                  static_cast<const float *>(ctx.target.data),
                  static_cast<float *>(ctx.loss.data),
                  static_cast<float *>(ctx.grad.data), actual_scale, m_delta);
  } else if (data_type == DataType::Float16) {
    if ((n & 1) == 0) {
      linear_kernel(huber_kernel_f16_h2, 0, to_cuda_stream(ctx.queue), n / 2,
                    reinterpret_cast<const __half2 *>(ctx.pred.data),
                    reinterpret_cast<const __half2 *>(ctx.target.data),
                    reinterpret_cast<__half2 *>(ctx.loss.data),
                    reinterpret_cast<__half2 *>(ctx.grad.data), actual_scale, m_delta);
    } else {
      linear_kernel(huber_kernel_f16, 0, to_cuda_stream(ctx.queue), n,
                    static_cast<const half *>(ctx.pred.data),
                    static_cast<const half *>(ctx.target.data),
                    static_cast<half *>(ctx.loss.data),
                    static_cast<half *>(ctx.grad.data), actual_scale, m_delta);
    }
  } else {
    throw std::runtime_error("HuberLoss: only float32/float16 are supported");
  }

}

} // namespace tinygs