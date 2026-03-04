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

/** @file   vec.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Tiny vector / matrix / quaternion implementation.
 */

#pragma once

#include <tinygs/common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace tinygs {

using vec2 = glm::fvec2;
using vec3 = glm::fvec3;
using vec4 = glm::fvec4;
using quat = glm::fquat;
using ivec3 = glm::ivec3;
using uvec3 = glm::uvec3;

using mat2x2 = glm::mat2x2;
using mat3x3 = glm::mat3x3;
using mat4x4 = glm::mat4x4;

TINYGS_HOST_DEVICE inline mat3x3 quat_to_mat3(quat q) {
  return glm::mat3_cast(q);
}

// Import external cwise functions into ngp namespace to avoid
// name resolution problems related to the vector-values versions defined below.
template <typename T> TINYGS_HOST_DEVICE T min(T a, T b) { return std::min(a, b); }
template <typename T> TINYGS_HOST_DEVICE T max(T a, T b) { return std::max(a, b); }
template <typename T> TINYGS_HOST_DEVICE T clamp(T a, T b, T c) { return a < b ? b : (c < a ? c : a); }
template <typename T> TINYGS_HOST_DEVICE T copysign(T a, T b) { return std::copysign(a, b); }
template <typename T> TINYGS_HOST_DEVICE T sign(T a) { return std::copysign((T)1, a); }
template <typename T> TINYGS_HOST_DEVICE T mix(T a, T b, T c) { return a * ((T)1 - c) + b * c; }
template <typename T> TINYGS_HOST_DEVICE T floor(T a) { return std::floor(a); }
template <typename T> TINYGS_HOST_DEVICE T round(T a) { return std::round(a); }
template <typename T> TINYGS_HOST_DEVICE T ceil(T a) { return std::ceil(a); }
template <typename T> TINYGS_HOST_DEVICE T abs(T a) { return std::abs(a); }
template <typename T> TINYGS_HOST_DEVICE T distance(T a, T b) { return std::abs(a - b); }
template <typename T> TINYGS_HOST_DEVICE T sin(T a) { return std::sin(a); }
template <typename T> TINYGS_HOST_DEVICE T asin(T a) { return std::asin(a); }
template <typename T> TINYGS_HOST_DEVICE T cos(T a) { return std::cos(a); }
template <typename T> TINYGS_HOST_DEVICE T acos(T a) { return std::acos(a); }
template <typename T> TINYGS_HOST_DEVICE T tan(T a) { return std::tan(a); }
template <typename T> TINYGS_HOST_DEVICE T atan(T a) { return std::atan(a); }
template <typename T> TINYGS_HOST_DEVICE T sqrt(T a) { return std::sqrt(a); }
template <typename T> TINYGS_HOST_DEVICE T exp(T a) { return std::exp(a); }
template <typename T> TINYGS_HOST_DEVICE T log(T a) { return std::log(a); }
template <typename T> TINYGS_HOST_DEVICE T exp2(T a) { return std::exp2(a); }
template <typename T> TINYGS_HOST_DEVICE T log2(T a) { return std::log2(a); }
template <typename T> TINYGS_HOST_DEVICE T pow(T a, T b) { return std::pow(a, b); }
template <typename T> TINYGS_HOST_DEVICE T isfinite(T a) {
#if defined(__CUDA_ARCH__)
	return ::isfinite(a);
#else
	return std::isfinite(a);
#endif
}



inline TINYGS_HOST_DEVICE float fma(float a, float b, float c) { return fmaf(a, b, c); }

#ifdef __CUDACC__
inline TINYGS_DEVICE __half fma(__half a, __half b, __half c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
	return __hfma(a, b, c);
#else
	return __float2half(fmaf(__half2float(a), __half2float(b), __half2float(c)));
#endif
}
#endif

#define CWISE_OP(operation, type_result, expr, ...)       \
  template <typename T, int N>                            \
  TINYGS_HOST_DEVICE type_result operation(__VA_ARGS__) { \
    type_result result;                                   \
    TINYGS_PRAGMA_UNROLL                                  \
    for (uint32_t i = 0; i < N; ++i) {                    \
      result[i] = expr;                                   \
    }                                                     \
    return result;                                        \
  }

#define TVEC glm::vec<N, T>
#define BVEC glm::vec<N, bool>

// CWISE_OP(operator+, TVEC, a[i] + b[i], const TVEC& a, const TVEC& b)
// CWISE_OP(operator+, TVEC, a + b[i], T a, const TVEC& b)
// CWISE_OP(operator+, TVEC, a[i] + b, const TVEC& a, T b)

// CWISE_OP(operator-, TVEC, a[i] - b[i], const TVEC& a, const TVEC& b)
// CWISE_OP(operator-, TVEC, a - b[i], T a, const TVEC& b)
// CWISE_OP(operator-, TVEC, a[i] - b, const TVEC& a, T b)

// CWISE_OP(operator*, TVEC, a[i] * b[i], const TVEC& a, const TVEC& b)
// CWISE_OP(operator*, TVEC, a * b[i], T a, const TVEC& b)
// CWISE_OP(operator*, TVEC, a[i] * b, const TVEC& a, T b)

// CWISE_OP(operator/, TVEC, a[i] / b[i], const TVEC& a, const TVEC& b)
// CWISE_OP(operator/, TVEC, a / b[i], T a, const TVEC& b)
// CWISE_OP(operator/, TVEC, a[i] / b, const TVEC& a, T b)


CWISE_OP(min, TVEC, min(a[i], b[i]), const TVEC& a, const TVEC& b)
CWISE_OP(min, TVEC, min(a[i], b), const TVEC& a, T b)
CWISE_OP(min, TVEC, min(a, b[i]), T a, const TVEC& b)

CWISE_OP(max, TVEC, max(a[i], b[i]), const TVEC& a, const TVEC& b)
CWISE_OP(max, TVEC, max(a[i], b), const TVEC& a, T b)
CWISE_OP(max, TVEC, max(a, b[i]), T a, const TVEC& b)

CWISE_OP(clamp, TVEC, clamp(a[i], b[i], c[i]), const TVEC& a, const TVEC& b, const TVEC& c)
CWISE_OP(clamp, TVEC, clamp(a[i], b[i], c), const TVEC& a, const TVEC& b, T c)
CWISE_OP(clamp, TVEC, clamp(a[i], b, c[i]), const TVEC& a, T b, const TVEC& c)
CWISE_OP(clamp, TVEC, clamp(a[i], b, c), const TVEC& a, T b, T c)

CWISE_OP(copysign, TVEC, copysign(a[i], b[i]), const TVEC& a, const TVEC& b)
CWISE_OP(copysign, TVEC, copysign(a[i], b), const TVEC& a, T b)
CWISE_OP(copysign, TVEC, copysign(a, b[i]), T a, const TVEC& b)

CWISE_OP(sign, TVEC, sign(a[i]), const TVEC& a)

CWISE_OP(mix, TVEC, a[i] * ((T)1 - c[i]) + b[i] * c[i], const TVEC& a, const TVEC& b, const TVEC& c)
CWISE_OP(mix, TVEC, a[i] * ((T)1 - c) + b[i] * c, const TVEC& a, const TVEC& b, T c)

// CWISE_OP(operator-, TVEC, -a[i], const TVEC& a)
CWISE_OP(floor, TVEC, floor(a[i]), const TVEC& a)
CWISE_OP(round, TVEC, round(a[i]), const TVEC& a)
CWISE_OP(ceil, TVEC, ceil(a[i]), const TVEC& a)
CWISE_OP(abs, TVEC, abs(a[i]), const TVEC& a)
CWISE_OP(sin, TVEC, sin(a[i]), const TVEC& a)
CWISE_OP(asin, TVEC, asin(a[i]), const TVEC& a)
CWISE_OP(cos, TVEC, cos(a[i]), const TVEC& a)
CWISE_OP(acos, TVEC, acos(a[i]), const TVEC& a)
CWISE_OP(tan, TVEC, tan(a[i]), const TVEC& a)
CWISE_OP(atan, TVEC, atan(a[i]), const TVEC& a)
CWISE_OP(sqrt, TVEC, sqrt(a[i]), const TVEC& a)
CWISE_OP(exp, TVEC, exp(a[i]), const TVEC& a)
CWISE_OP(log, TVEC, log(a[i]), const TVEC& a)
CWISE_OP(exp2, TVEC, exp2(a[i]), const TVEC& a)
CWISE_OP(log2, TVEC, log2(a[i]), const TVEC& a)
CWISE_OP(pow, TVEC, pow(a[i], b), const TVEC& a, T b)
CWISE_OP(pow, TVEC, pow(a[i], b[i]), const TVEC& a, const TVEC& b)
CWISE_OP(isfinite, BVEC, isfinite(a[i]), const TVEC& a)

#undef CWISE_OP

#define INPLACE_OP(operation, type_b, expr)               \
  template <typename T, int N>             \
  TINYGS_HOST_DEVICE TVEC& operation(TVEC& a, type_b b) { \
    TINYGS_PRAGMA_UNROLL                                  \
    for (uint32_t i = 0; i < N; ++i) {                    \
      expr;                                               \
    }                                                     \
    return a;                                             \
  }

// INPLACE_OP(operator*=, const TVEC&, a[i] *= b[i])
// INPLACE_OP(operator/=, const TVEC&, a[i] /= b[i])
// INPLACE_OP(operator+=, const TVEC&, a[i] += b[i])
// INPLACE_OP(operator-=, const TVEC&, a[i] -= b[i])

// INPLACE_OP(operator*=, T, a[i] *= b)
// INPLACE_OP(operator/=, T, a[i] /= b)

#define REDUCTION_OP(operation, type_result, init, expr, ...) \
  template <typename T, int N>                 \
  TINYGS_HOST_DEVICE type_result operation(__VA_ARGS__) {     \
    type_result result = init;                                \
    TINYGS_PRAGMA_UNROLL                                      \
    for (uint32_t i = 0; i < N; ++i) {                        \
      expr;                                                   \
    }                                                         \
    return result;                                            \
  }

REDUCTION_OP(dot,     T, (T)0, result += a[i] * b[i], const TVEC& a, const TVEC& b)
REDUCTION_OP(sum,     T, (T)0, result += a[i], const TVEC& a)
REDUCTION_OP(mean,    T, (T)0, result += a[i] / (T)N, const TVEC& a)
REDUCTION_OP(product, T, (T)1, result *= a[i], const TVEC& a)
REDUCTION_OP(min,     T, (T)std::numeric_limits<T>::infinity(), result = min(result, a[i]), const TVEC& a)
REDUCTION_OP(max,     T, (T)-std::numeric_limits<T>::infinity(), result = max(result, a[i]), const TVEC& a)
REDUCTION_OP(length2, T, (T)0, result += a[i] * a[i], const TVEC& a)

#undef REDUCTION_OP

#define BOOL_REDUCTION_OP(operation, type_result, init, expr, ...) \
  template <int N>                                                 \
  TINYGS_HOST_DEVICE type_result operation(__VA_ARGS__) {          \
    type_result result = init;                                     \
    TINYGS_PRAGMA_UNROLL                                           \
    for (uint32_t i = 0; i < N; ++i) {                             \
      expr;                                                        \
    }                                                              \
    return result;                                                 \
  }

BOOL_REDUCTION_OP(all, bool, true, result &= a[i], const BVEC& a)
BOOL_REDUCTION_OP(any, bool, false, result |= a[i], const BVEC& a)

#undef BOOL_REDUCTION_OP

template <typename T, int N>
TINYGS_HOST_DEVICE T length(const TVEC& a) {
	return std::sqrt(length2(a));
}

template <typename T, int N>
TINYGS_HOST_DEVICE T distance(const TVEC& a, const TVEC& b) {
	return length(a - b);
}

template <typename T, int N>
TINYGS_HOST_DEVICE TVEC normalize(const TVEC& v) {
	T len = length(v);
	if (len <= (T)0) {
		TVEC result{(T)0};
		result[0] = (T)1;
		return result;
	}
	return v / len;
}


#if defined(__CUDACC__)
inline TINYGS_DEVICE void atomic_add_gmem_float(float* addr, float in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
	int in_int = *((int*)&in);
	asm ("red.relaxed.gpu.global.add.f32 [%0], %1;" :: "l"(addr), "r"(in_int));
#else
	atomicAdd(addr, in);
#endif
}

template <typename T, int N>
TINYGS_DEVICE void atomic_add(T* dst, const glm::vec<N, T>& a) {
	TINYGS_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		atomicAdd(dst + i, a[i]);
	}
}

template <int N>
TINYGS_DEVICE void atomic_add_gmem(float* dst, const glm::vec<N, float>& a) {
	TINYGS_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N; ++i) {
		atomic_add_gmem_float(dst + i, a[i]);
	}
}
#endif

}