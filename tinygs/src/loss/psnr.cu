#include "tinygs/cuda/common_device.cuh"
#include "tinygs/loss/psnr.hpp"
#include "tinygs/cuda/reduce.hpp"
#include <cuda_fp16.h>
#include <cmath>


// FastGS reference (image_utils.py):
//   def psnr(img1, img2):
//       mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
//       return 20 * torch.log10(1.0 / torch.sqrt(mse))
//
// Note: FastGS computes MSE per channel, then PSNR per channel, then averages.
// This is NOT the same as computing MSE over all pixels and then PSNR.

/**
 * Computes Peak Signal-to-Noise Ratio (PSNR) per-channel and averages.
 *
 * For each channel c: PSNR_c = -10 * log10(MSE_c)
 * Final PSNR = mean(PSNR_r, PSNR_g, PSNR_b)
 *
 * This matches the FastGS reference implementation where:
 *   mse = view(img.shape[0], -1).mean(1)  # per-channel MSE
 *   psnr = 20 * log10(1.0 / sqrt(mse))     # per-channel PSNR
 *   result = psnr.mean()                    # average across channels
 *
 * For normalized images in [0, 1], MAX_I = 1.0, so:
 *   20 * log10(1.0 / sqrt(MSE)) = 20 * log10(1) - 10 * log10(MSE) = -10 * log10(MSE)
 */

// CUDA kernel to compute squared differences (float32)
__global__ void psnr_squared_diff_kernel(int N, const float *__restrict__ pred,
                                         const float *__restrict__ target,
                                         float *__restrict__ squared_diff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const float p = tinygs::saturate(pred[i]);
  const float t = target[i];
  const float diff = p - t;
  squared_diff[i] = diff * diff;
}

// fp16 scalar specialization
__global__ void psnr_squared_diff_kernel_f16(int N, const half *__restrict__ pred,
                                             const half *__restrict__ target,
                                             float *__restrict__ squared_diff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  const float p = tinygs::saturate(__half2float(pred[i]));
  const float t = __half2float(target[i]);
  const float diff = p - t;
  squared_diff[i] = diff * diff;
}

// fp16 half2 SIMD specialization (writes two float outputs per thread)
__global__ void psnr_squared_diff_kernel_f16_h2(int N_pairs, const __half2 *__restrict__ pred,
                                                const __half2 *__restrict__ target,
                                                float *__restrict__ squared_diff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_pairs) {
    return;
  }
  const float2 p_raw = __half22float2(pred[i]);
  const float2 t = __half22float2(target[i]);
  const float p0 = tinygs::saturate(p_raw.x);
  const float p1 = tinygs::saturate(p_raw.y);
  const float diff0 = p0 - t.x;
  const float diff1 = p1 - t.y;
  squared_diff[2 * i + 0] = diff0 * diff0;
  squared_diff[2 * i + 1] = diff1 * diff1;
}

namespace tinygs {

float PsnrMetric::evaluate(Image pred, Image target) {
  // Total elements in padded memory layout (includes padding, which is zero-initialized)
  int n = pred.shape.padded_size();
  // Stride between channels in the tiled memory layout
  uint32_t channel_stride = pred.shape.padded_width() * pred.shape.padded_height();
  // Number of valid pixels per channel (excluding padding)
  uint32_t pixels_per_channel = pred.shape.width * pred.shape.height;
  uint32_t channels = pred.shape.channel;
  
  if (pred.shape != target.shape) {
    throw std::runtime_error(fmt::format(
      "Prediction and target shapes must match, got: {} vs {}", 
      to_string(pred.shape), to_string(target.shape)));
  }
  if (pred.data_type != target.data_type) {
    throw std::runtime_error(fmt::format(
        "PSNR: prediction and target data type must be the same, got {} vs. {}",
        to_string(pred.data_type), to_string(target.data_type)));
  }

  // Allocate temporary memory for squared differences
  if (m_sqr_diff.size() < n) {
    m_sqr_diff = GPUBuffer<float>(n);
  }
  m_sqr_diff.memset(0);

  float* squared_diff = m_sqr_diff.data();
  // Compute squared differences
  if (pred.data_type == DataType::Float32) {
    linear_kernel(psnr_squared_diff_kernel, 0, nullptr, n,
      static_cast<const float*>(pred.data),
      static_cast<const float*>(target.data),
      squared_diff);
  } else if (pred.data_type == DataType::Float16) {
    if ((n & 1) == 0) {
      linear_kernel(psnr_squared_diff_kernel_f16_h2, 0, nullptr, n / 2,
        reinterpret_cast<const __half2*>(pred.data),
        reinterpret_cast<const __half2*>(target.data),
        squared_diff);
    } else {
      linear_kernel(psnr_squared_diff_kernel_f16, 0, nullptr, n,
        static_cast<const half*>(pred.data),
        static_cast<const half*>(target.data),
        squared_diff);
    }
  } else {
    throw std::runtime_error("PSNR: only float32/float16 are supported");
  }

  // Compute per-channel PSNR and average
  // This matches the FastGS reference implementation:
  //   PSNR = mean([-10*log10(MSE_r), -10*log10(MSE_g), -10*log10(MSE_b)])
  // Note: Padding elements are zero-initialized, so they don't affect the sum.
  //       We sum over channel_stride elements (including padding zeros), then
  //       divide by pixels_per_channel (actual valid pixels).
  float psnr_sum = 0.0f;
  for (uint32_t c = 0; c < channels; ++c) {
    float sum_c = gpu_sum(squared_diff + c * channel_stride, channel_stride);
    float mse_c = sum_c / pixels_per_channel;
    // For normalized images, MAX_I = 1.0, so PSNR = -10 * log10(MSE)
    if (mse_c == 0.0f) {
      psnr_sum += 100.0f;  // Perfect match for this channel
    } else {
      psnr_sum += -10.0f * log10f(mse_c);
    }
  }
  return psnr_sum / static_cast<float>(channels);
}

} // namespace tinygs