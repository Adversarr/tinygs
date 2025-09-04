#include "tinygs/random/multinomial.hpp"

#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cstdio>

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
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_cdf, d_cdf, K);
    auto b_temp = GPUBuffer<uint8_t>(stream, temp_bytes);
    d_temp = b_temp.data();
    CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_cdf, d_cdf, K));

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
    CUDA_CHECK_THROW(cudaPeekAtLastError());

    return b_out;
}
}