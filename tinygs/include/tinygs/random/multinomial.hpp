#pragma once

#include <tinygs/cuda/gpu_memory.hpp>
#include <vector>
#include <memory>

namespace tinygs {

/// @brief CUDA implementation of multinomial sampling (with replacement)
/// @param d_weights Array of non-negative weights on GPU
/// @param K Number of categories
/// @param num_samples Number of samples to draw
/// @param seed Random seed
GPUBuffer<int> multinomial_cuda_with_replacement(
    const float* d_weights,
    int K,
    int num_samples,
    int seed,
    cudaStream_t stream = nullptr
);

/// @brief CPU implementation of multinomial sampling (with replacement)
/// @param weights Array of non-negative weights
/// @param K Number of categories
/// @param num_samples Number of samples to draw
/// @param seed Random seed
std::vector<int> multinomial_cpu_with_replacement(
    const float* weights,
    int K,
    int num_samples,
    int seed
);

/// @brief CPU implementation of multinomial sampling (with replacement) on GPU
GPUBuffer<int> multinomial_cuda_cpu(
    const float* d_weights,
    int K,
    int num_samples,
    int seed,
    cudaStream_t stream = nullptr
);

/// @brief CPU implementation of multinomial sampling (without replacement) on GPU
/// Selects indices proportionally to weights, without replacement, using
/// Efraimidis–Spirakis PPS sampling via keys u^{1/w} and taking top-M.
GPUBuffer<int> multinomial_cuda_cpu_without_replacement(
    const float* d_weights,
    int K,
    int num_samples,
    int seed,
    cudaStream_t stream = nullptr
);

/// @brief CPU implementation of multinomial sampling (without replacement)
/// Selects indices proportionally to weights, without replacement.
std::vector<int> multinomial_cpu_without_replacement(
    const float* weights,
    int K,
    int num_samples,
    int seed
);

}