#include "tinygs/cuda/common_device.cuh"
#include "tinygs/loss/psnr.hpp"
#include "tinygs/cuda/reduce.hpp"
#include <cuda_fp16.h>
#include <cmath>


// def psnr(img1, img2):
//     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
//     return 20 * torch.log10(1.0 / torch.sqrt(mse))

/**
 * Computes Peak Signal-to-Noise Ratio (PSNR).
 *
 * PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
 * where MAX_I is the maximum possible pixel value (typically 1.0 for normalized images)
 * and MSE = mean((pred - target)^2)
 *
 * Since we're working with normalized images in the range [0, 1], MAX_I = 1.0
 * and log10(1.0) = 0, so PSNR = -10 * log10(MSE)
 */

// CUDA kernel to compute squared differences
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

namespace tinygs {

float PsnrMetric::evaluate(Image pred, Image target) {
  int n = pred.shape.padded_size(); // physical
  int npix = pred.shape.size();     // actual
  
  // Allocate temporary memory for squared differences
  if (m_sqr_diff.size() < n) {
    m_sqr_diff = GPUBuffer<float>(n);
  }
  float* squared_diff = m_sqr_diff.data();
  
  // Compute squared differences
  linear_kernel(psnr_squared_diff_kernel, 0, nullptr, n,
    static_cast<const float*>(pred.data),
    static_cast<const float*>(target.data),
    squared_diff);

  // Compute MSE (mean squared error)
  float mse = gpu_sum(squared_diff, n) / npix;

  // Compute PSNR
  // For normalized images, MAX_I = 1.0, so 20 * log10(MAX_I) = 0
  // PSNR = -10 * log10(MSE)
  if (mse == 0.0f) {
    // Avoid log(0) which would be -inf
    return 100.0f; // Return a large value for perfect match
  }
  
  return -10.0f * log10f(mse);
}

} // namespace tinygs