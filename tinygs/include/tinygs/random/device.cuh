/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   random.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Collection of CUDA kernels related to random numbers
 */

#pragma once

#include "tinygs/common.hpp"
#include "tinygs/cuda/common_device.cuh"
#include "tinygs/random/pcg32.hpp"

namespace tinygs {

template <typename T, typename RNG, size_t N_TO_GENERATE, typename F>
__global__ void generate_random_kernel(const size_t n_elements, RNG rng, T* __restrict__ out, const F transform) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t n_threads = blockDim.x * gridDim.x;

	rng.advance(i*N_TO_GENERATE);

	TINYGS_PRAGMA_UNROLL
	for (size_t j = 0; j < N_TO_GENERATE; ++j) {
		const size_t idx = i + n_threads * j;
		if (idx >= n_elements) {
			return;
		}

		out[idx] = transform((T)rng.next_float());
	}
}

template <typename T, typename RNG, typename F>
void generate_random(cudaStream_t stream, RNG& rng, size_t n_elements, T* out, F&& transform) {
	static constexpr size_t N_TO_GENERATE = 4;

	size_t n_threads = div_round_up(n_elements, N_TO_GENERATE);
	generate_random_kernel<T, RNG, N_TO_GENERATE><<<n_blocks_linear(n_threads), N_THREADS_LINEAR, 0, stream>>>(n_elements, rng, out, transform);

	rng.advance(n_elements);
}

template <typename T, typename RNG>
void generate_random_uniform(cudaStream_t stream, RNG& rng, size_t n_elements, T* out, const T lower = (T)0.0, const T upper = (T)1.0) {
	generate_random(stream, rng, n_elements, out, [upper, lower] __device__ (T val) { return val * (upper - lower) + lower; });
}

template <typename T, typename RNG>
void generate_random_uniform(RNG& rng, size_t n_elements, T* out, const T lower = (T)0.0, const T upper = (T)1.0) {
	generate_random_uniform(nullptr, rng, n_elements, out, lower, upper);
}

template <typename T, typename RNG>
void generate_random_logistic(cudaStream_t stream, RNG& rng, size_t n_elements, T* out, const T mean = (T)0.0, const T stddev = (T)1.0) {
	generate_random(stream, rng, n_elements, out, [mean, stddev] __device__ (T val) { return (T)logit(val) * stddev * 0.551328895f + mean; });
}

template <typename T, typename RNG>
void generate_random_logistic(RNG& rng, size_t n_elements, T* out, const T mean = (T)0.0, const T stddev = (T)1.0) {
	generate_random_logistic(nullptr, rng, n_elements, out, mean, stddev);
}

/// @brief CUDA kernel that generates normally-distributed random numbers via Box-Muller.
///        Each thread produces a pair of normals from two uniform samples.
template <typename T, typename RNG, size_t N_PAIRS>
__global__ void generate_random_normal_kernel(
    const size_t n_elements, RNG rng, T* __restrict__ out, const T mean, const T stddev) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t n_threads = blockDim.x * gridDim.x;

	// Each thread consumes 2*N_PAIRS uniform samples and writes 2*N_PAIRS normals
	rng.advance(i * N_PAIRS * 2);

	TINYGS_PRAGMA_UNROLL
	for (size_t p = 0; p < N_PAIRS; ++p) {
		const size_t idx0 = i + n_threads * (2 * p);
		const size_t idx1 = i + n_threads * (2 * p + 1);

		// Clamp to (0,1) to avoid log(0)
		T u1 = fmaxf((T)rng.next_float(), (T)1e-7);
		T u2 = (T)rng.next_float();

		T r    = sqrtf((T)-2.0 * logf(u1));
		T theta = (T)(2.0 * M_PI) * u2;
		T z0 = r * cosf(theta);
		T z1 = r * sinf(theta);

		if (idx0 < n_elements) out[idx0] = z0 * stddev + mean;
		if (idx1 < n_elements) out[idx1] = z1 * stddev + mean;
	}
}

/// @brief Generate normally-distributed (Gaussian) random numbers via Box-Muller transform.
///        Matches the semantics of torch.normal(mean, std).
template <typename T, typename RNG>
void generate_random_normal(cudaStream_t stream, RNG& rng, size_t n_elements, T* out,
                            const T mean = (T)0.0, const T stddev = (T)1.0) {
	static constexpr size_t N_PAIRS = 2;  // 4 outputs per thread (2 pairs)
	size_t n_threads = div_round_up(n_elements, N_PAIRS * 2);
	generate_random_normal_kernel<T, RNG, N_PAIRS>
		<<<n_blocks_linear(n_threads), N_THREADS_LINEAR, 0, stream>>>(
			n_elements, rng, out, mean, stddev);
	rng.advance(n_elements);  // Advance by the number of elements produced
}

/// @brief Convenience overload: generate normally-distributed random numbers on the default stream.
template <typename T, typename RNG>
void generate_random_normal(RNG& rng, size_t n_elements, T* out,
                            const T mean = (T)0.0, const T stddev = (T)1.0) {
	generate_random_normal(nullptr, rng, n_elements, out, mean, stddev);
}

/// @brief Generate random integers using a custom transformation function
template <typename RNG, size_t N_TO_GENERATE>
__global__ void generate_random_int_kernel(const size_t n_elements, RNG rng, uint32_t* __restrict__ out, const uint32_t bound) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const size_t n_threads = blockDim.x * gridDim.x;

	rng.advance(i*N_TO_GENERATE);

	TINYGS_PRAGMA_UNROLL
	for (size_t j = 0; j < N_TO_GENERATE; ++j) {
		const size_t idx = i + n_threads * j;
		if (idx >= n_elements) {
			return;
		}

		out[idx] = rng.next_uint(bound);
	}
}

/// @brief Generate random integers using a custom transformation function
template <typename RNG>
void generate_random_ui32(cudaStream_t stream, RNG& rng, size_t n_elements, uint32_t* out, uint32_t bound) {
	static constexpr size_t N_TO_GENERATE = 4;
	size_t n_threads = div_round_up(n_elements, N_TO_GENERATE);
	generate_random_int_kernel<RNG, N_TO_GENERATE>
			<<<n_blocks_linear(n_threads), N_THREADS_LINEAR, 0, stream>>>(
					n_elements, rng, out, bound);

	rng.advance(n_elements);
}
}
