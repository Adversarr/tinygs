#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#ifndef TINYGS_HOST_DEVICE
#define TINYGS_HOST_DEVICE
#endif

#ifndef TINYGS_PRAGMA_UNROLL
#define TINYGS_PRAGMA_UNROLL
#endif

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

// Import scalar cwise functions into tinygs namespace to avoid name resolution
// problems related to vector overloads defined below.
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

#define REDUCTION_OP(operation, type_result, init, expr, ...) \
  template <typename T, int N>                                 \
  TINYGS_HOST_DEVICE type_result operation(__VA_ARGS__) {      \
    type_result result = init;                                 \
    TINYGS_PRAGMA_UNROLL                                       \
    for (uint32_t i = 0; i < N; ++i) {                         \
      expr;                                                    \
    }                                                          \
    return result;                                             \
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
  template <int N>                                                  \
  TINYGS_HOST_DEVICE type_result operation(__VA_ARGS__) {           \
    type_result result = init;                                      \
    TINYGS_PRAGMA_UNROLL                                            \
    for (uint32_t i = 0; i < N; ++i) {                              \
      expr;                                                         \
    }                                                               \
    return result;                                                  \
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

} // namespace tinygs
