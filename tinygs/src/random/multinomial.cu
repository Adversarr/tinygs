#include "tinygs/random/multinomial.hpp"

#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <numeric>
#include <algorithm>
#include "tinygs/random/pcg32.hpp"

namespace tinygs{

__device__ __forceinline__ int lower_bound_cdf(const float* cdf, int K, float u) {
    int lo = 0, hi = K - 1;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (u <= cdf[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

__global__ void multinomial_sample_kernel(
    const float* __restrict__ cdf,
    int K,
    float total_sum,
    int num_samples,
    int seed,
    int* __restrict__ out_indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);

    float r = curand_uniform(&state);
    float u = r * total_sum;
    if (u >= total_sum) {
        u = nextafterf(total_sum, 0.0f);
    }

    int idx = lower_bound_cdf(cdf, K, u);
    out_indices[i] = idx;
}

GPUBuffer<int> multinomial_cuda_with_replacement(
  const float* d_weights, 
  int K, 
  int num_samples, 
  int seed,
  cudaStream_t stream)
{
  if (K <= 0 || num_samples <= 0) {
      throw std::runtime_error(fmt::format("Invalid K={} or num_samples={}", K, num_samples));
  }

  auto b_cdf = GPUBuffer<float>(stream, K);
  float* d_cdf = b_cdf.data();
  CUDA_CHECK_THROW(cudaMemcpyAsync(d_cdf, d_weights, sizeof(float) * K, cudaMemcpyDeviceToDevice, stream));

  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_cdf, d_cdf, K, stream));
  auto b_temp = GPUBuffer<uint8_t>(stream, temp_bytes);
  d_temp = b_temp.data();
  CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_cdf, d_cdf, K, stream));

  float h_total = 0.0f;
  CUDA_CHECK_THROW(cudaMemcpyAsync(&h_total, d_cdf + (K - 1), sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

  if (!(h_total > 0.0f) || !isfinite(h_total)) {
      throw std::runtime_error(fmt::format("Invalid weights: h_total={:.4e}", h_total));
  }

  auto b_out = GPUBuffer<int>(stream, num_samples);
  int* d_out = b_out.data();

  int threads = 256;
  int blocks = (num_samples + threads - 1) / threads;
  multinomial_sample_kernel<<<blocks, threads, 0, stream>>>(d_cdf, K, h_total, num_samples, seed, d_out);
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
  return b_out;
}

// CPU implementation of lower_bound_cdf
int lower_bound_cdf_cpu(const float *cdf, int K, float u) {
  int lo = 0, hi = K - 1;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    if (u <= cdf[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

std::vector<int> multinomial_cpu_with_replacement(
  const float* weights,
  int K,
  int num_samples,
  int seed)
{
  if (K <= 0 || num_samples <= 0) {
    throw std::runtime_error(
        fmt::format("Invalid K={} or num_samples={}", K, num_samples));
  }

  // Compute CDF on CPU
  std::vector<float> cdf(K);
  std::partial_sum(weights, weights + K, cdf.begin());

  float total_sum = cdf[K - 1];

  if (!(total_sum > 0.0f) || !std::isfinite(total_sum)) {
    throw std::runtime_error(
        fmt::format("Invalid weights: total_sum={:.4e}", total_sum));
  }

  // Initialize random number generator
  pcg32 rng(seed);

  // Sample indices
  std::vector<int> out_indices(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    float r = rng.next_float();
    float u = r * total_sum;
    if (u >= total_sum) {
      u = std::nextafter(total_sum, 0.0f);
    }

    int idx = lower_bound_cdf_cpu(cdf.data(), K, u);
    out_indices[i] = idx;
  }

  return out_indices;
}

GPUBuffer<int> multinomial_cuda_cpu(
  const float* d_weights,
  int K,
  int num_samples,
  int seed,
  cudaStream_t stream)
{
  std::vector<float> h_weights(K);
  CUDA_CHECK_THROW(cudaMemcpyAsync(h_weights.data(), d_weights, sizeof(float) * K, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

  auto h_out = multinomial_cpu_with_replacement(h_weights.data(), K, num_samples, seed);
  auto b_out = GPUBuffer<int>(stream, num_samples);
  CUDA_CHECK_THROW(cudaMemcpyAsync(b_out.data(), h_out.data(), sizeof(int) * num_samples, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
  return b_out;
}

// CPU sampling without replacement using Efraimidis–Spirakis PPS scheme
std::vector<int> multinomial_cpu_without_replacement(
  const float* weights,
  int K,
  int num_samples,
  int seed)
{
  if (K <= 0 || num_samples <= 0) {
    throw std::runtime_error(
        fmt::format("Invalid K={} or num_samples={}", K, num_samples));
  }

  pcg32 rng(seed);
  struct KeyIdx { float key; int idx; };
  std::vector<KeyIdx> keys;
  keys.reserve(K);

  float total_pos = 0.0f;
  for (int i = 0; i < K; ++i) {
    float w = weights[i];
    if (w > 0.0f && std::isfinite(w)) {
      total_pos += w;
      float u = rng.next_float();
      // Ensure u in (0,1) open interval to avoid edge cases
      if (u <= 0.0f) u = std::nextafter(0.0f, 1.0f);
      if (u >= 1.0f) u = std::nextafter(1.0f, 0.0f);
      float key = std::pow(u, 1.0f / w);
      keys.push_back({key, i});
    }
  }

  if (total_pos <= 0.0f || keys.empty()) {
    throw std::runtime_error("Invalid weights: no positive entries for sampling without replacement");
  }

  if (num_samples > static_cast<int>(keys.size())) {
    num_samples = static_cast<int>(keys.size());
  }

  // Select top-M entries by key
  std::nth_element(keys.begin(), keys.begin() + num_samples, keys.end(),
                   [](const KeyIdx& a, const KeyIdx& b) { return a.key > b.key; });
  std::sort(keys.begin(), keys.begin() + num_samples,
            [](const KeyIdx& a, const KeyIdx& b) { return a.key > b.key; });

  std::vector<int> out(num_samples);
  for (int i = 0; i < num_samples; ++i) out[i] = keys[i].idx;
  return out;
}

GPUBuffer<int> multinomial_cuda_cpu_without_replacement(
  const float* d_weights,
  int K,
  int num_samples,
  int seed,
  cudaStream_t stream)
{
  std::vector<float> h_weights(K);
  CUDA_CHECK_THROW(cudaMemcpyAsync(h_weights.data(), d_weights, sizeof(float) * K, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

  auto h_out = multinomial_cpu_without_replacement(h_weights.data(), K, num_samples, seed);
  auto b_out = GPUBuffer<int>(stream, num_samples);
  CUDA_CHECK_THROW(cudaMemcpyAsync(b_out.data(), h_out.data(), sizeof(int) * num_samples, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
  return b_out;
}

}