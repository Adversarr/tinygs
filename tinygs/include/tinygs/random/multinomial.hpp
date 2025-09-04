#pragma once

#include <tinygs/cuda/gpu_memory.hpp>
namespace tinygs {

/**
 * @brief CUDA implementation of multinomial sampling (with replacement)
 * 
 * @param d_weights Array of non-negative weights of length K on GPU (normalization not required)
 * @param K Number of categories
 * @param num_samples Number of samples to draw
 * @param seed Random seed
 * @return int* Array of sampling results of length num_samples on GPU, caller is responsible for cudaFree
 */
GPUBuffer<int> multinomial_cuda_with_replacement(
    const float* d_weights, 
    int K, 
    int num_samples, 
    int seed,
    cudaStream_t stream = 0
);

}