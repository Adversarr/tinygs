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

/** @file   common_host.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#pragma once

#include <functional>
#include <tinygs/common.hpp>
#include <tinygs/platform/backend_types.hpp>
// #include <tinygs/cpp_api.h>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tinygs {

using namespace fmt::literals;

enum class LogSeverity {
  Info,
  Debug,
  Warning,
  Error,
  Success,
};

const std::function<void(LogSeverity, const std::string&)>& log_callback();
void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& callback);

template <typename... Ts>
void log(LogSeverity severity, fmt::format_string<Ts...> msg, Ts&&... args) {
		log_callback()(severity, fmt::format(msg, std::forward<Ts>(args)...));
}

inline void log(LogSeverity severity, const std::string& msg) {
	log_callback()(severity, msg);
}

// template <typename... Ts> void log_info(const std::string& msg, Ts&&... args) { log(LogSeverity::Info, msg, std::forward<Ts>(args)...); }
// template <typename... Ts> void log_debug(const std::string& msg, Ts&&... args) { log(LogSeverity::Debug, msg, std::forward<Ts>(args)...); }
// template <typename... Ts> void log_warning(const std::string& msg, Ts&&... args) { log(LogSeverity::Warning, msg, std::forward<Ts>(args)...); }
// template <typename... Ts> void log_error(const std::string& msg, Ts&&... args) { log(LogSeverity::Error, msg, std::forward<Ts>(args)...); }
// template <typename... Ts> void log_success(const std::string& msg, Ts&&... args) { log(LogSeverity::Success, msg, std::forward<Ts>(args)...); }
#define log_info(...) SPDLOG_INFO(__VA_ARGS__)
#define log_debug(...) SPDLOG_DEBUG(__VA_ARGS__)
#define log_warning(...) SPDLOG_WARN(__VA_ARGS__)
#define log_error(...) SPDLOG_ERROR(__VA_ARGS__)
#define log_success(...) SPDLOG_TRACE(__VA_ARGS__)

bool verbose();
void set_verbose(bool verbose);

#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error{FILE_LINE " check failed: " #x}; } while(0)

/// Check CUDA driver API call and throw on failure
#define CU_CHECK_THROW(x) \
	do { \
		CUresult _result = x; \
		if (_result != CUDA_SUCCESS) { \
			const char *msg; \
			cuGetErrorName(_result, &msg); \
			throw std::runtime_error{fmt::format(FILE_LINE " " #x " failed: {}", msg)}; \
		} \
	} while(0)

/// Check CUDA driver API call and print error on failure
#define CU_CHECK_PRINT(x) \
	do { \
		CUresult _result = x; \
		if (_result != CUDA_SUCCESS) { \
			const char *msg; \
			cuGetErrorName(_result, &msg); \
			log_error(FILE_LINE " " #x " failed: {}", msg); \
		} \
	} while(0)

/// Check CUDA runtime API call and throw on failure
#define CUDA_CHECK_THROW(x) \
	do { \
		cudaError_t _result = x; \
		if (_result != cudaSuccess) \
			throw std::runtime_error{fmt::format(FILE_LINE " " #x " failed: {}", cudaGetErrorString(_result))}; \
	} while(0)

/// Check CUDA runtime API call and print error on failure
#define CUDA_CHECK_PRINT(x) \
	do { \
		cudaError_t _result = x; \
		if (_result != cudaSuccess) \
			log_error(FILE_LINE " " #x " failed: {}", cudaGetErrorString(_result)); \
	} while(0)


//////////////////
// Misc helpers //
//////////////////

struct MemoryInfo {
	size_t total;
	size_t free;
	size_t used;
};

int cuda_runtime_version();
int cuda_device();
void set_cuda_device(int device);
int cuda_device_count();
bool cuda_supports_virtual_memory(int device);
std::string cuda_device_name(int device);
uint32_t cuda_compute_capability(int device);
uint32_t cuda_max_supported_compute_capability();
uint32_t cuda_supported_compute_capability(int device);
size_t cuda_max_shmem(int device);
uint32_t cuda_max_registers(int device);
size_t cuda_memory_granularity(int device);
MemoryInfo cuda_memory_info();

inline std::string cuda_runtime_version_string() {
  int v = cuda_runtime_version();
  return fmt::format("{}.{}", v / 1000, (v % 100) / 10);
}

inline bool cuda_supports_virtual_memory() {
  return cuda_supports_virtual_memory(cuda_device());
}

inline std::string cuda_device_name() {
  return cuda_device_name(cuda_device());
}

inline uint32_t cuda_compute_capability() {
  return cuda_compute_capability(cuda_device());
}

inline uint32_t cuda_supported_compute_capability() {
  return cuda_supported_compute_capability(cuda_device());
}

inline size_t cuda_max_shmem() {
  return cuda_max_shmem(cuda_device());
}

inline uint32_t cuda_max_registers() {
  return cuda_max_registers(cuda_device());
}

inline size_t cuda_memory_granularity() {
  return cuda_memory_granularity(cuda_device());
}

/// @brief Check if current CUDA device supports required features
void check_features_supported();

// Hash helpers taken from https://stackoverflow.com/a/50978188
template <typename T>
T xorshift(T n, int i) {
  return n ^ (n >> i);
}

inline uint32_t distribute(uint32_t n) {
  uint32_t p = 0x55555555ul;  // pattern of alternating 0 and 1
  uint32_t c = 3423571495ul;  // random uneven integer constant;
  return c * xorshift(p * xorshift(n, 16), 16);
}

inline uint64_t distribute(uint64_t n) {
  uint64_t p = 0x5555555555555555ull;    // pattern of alternating 0 and 1
  uint64_t c = 17316035218449499591ull;  // random uneven integer constant;
  return c * xorshift(p * xorshift(n, 32), 32);
}


template <typename T, typename S>
constexpr typename std::enable_if<std::is_unsigned<T>::value, T>::type rotl(const T n, const S i) {
	const T m = (std::numeric_limits<T>::digits - 1);
	const T c = i & m;
	return (n << c) | (n >> (((T)0 - c) & m)); // this is usually recognized by the compiler to mean rotation
}

template <typename T>
size_t hash_combine(std::size_t seed, const T& v) {
	return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute(std::hash<T>{}(v));
}

template <typename T>
std::string join(const T& components, const std::string& delim) {
	std::ostringstream s;
	for (const auto& component : components) {
		if (&components[0] != &component) {
			s << delim;
		}
		s << component;
	}

	return s.str();
}

std::string to_lower(std::string str);
std::string to_upper(std::string str);
inline bool equals_case_insensitive(const std::string& str1, const std::string& str2) {
	return to_lower(str1) == to_lower(str2);
}

inline std::string bytes_to_string(size_t bytes) {
  std::array<std::string, 7> suffixes = {{"B", "KB", "MB", "GB", "TB", "PB", "EB"}};

  double count = static_cast<double>(bytes);
  uint32_t i = 0;
	for (; (i + 1) < suffixes.size() && count >= 1024.0; ++i) {
    count /= 1024;
  }

  std::ostringstream oss;
	oss << std::fixed;
  oss.precision(3);
  oss << count << " " << suffixes[i];
  return oss.str();
}


class ScopeGuard {
public:
	ScopeGuard() = default;
	explicit ScopeGuard(const std::function<void()>& callback) : m_callback{callback} {}
	explicit ScopeGuard(std::function<void()>&& callback) : m_callback{std::move(callback)} {}
	ScopeGuard& operator=(const ScopeGuard& other) = delete;
	ScopeGuard(const ScopeGuard& other) = delete;
	ScopeGuard& operator=(ScopeGuard&& other) { std::swap(m_callback, other.m_callback); return *this; }
	ScopeGuard(ScopeGuard&& other) { *this = std::move(other); }
	~ScopeGuard() { if (m_callback) { m_callback(); } }

	void disarm() {
		m_callback = {};
	}
private:
	std::function<void()> m_callback;
};

inline cudaStream_t to_cuda_stream(BackendStream stream) {
  return reinterpret_cast<cudaStream_t>(stream);
}

inline BackendStream to_backend_stream(cudaStream_t stream) {
  return reinterpret_cast<BackendStream>(stream);
}

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))

template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
	if (n_elements <= 0) {
		return;
	}
	kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);
}

template <typename F>
__global__ void parallel_for_kernel(const size_t n_elements, F fun) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	fun(i);
}

template <typename F>
inline void parallel_for_gpu(uint32_t shmem_size, cudaStream_t stream, size_t n_elements, F&& fun) {
	if (n_elements <= 0) {
		return;
	}
	parallel_for_kernel<F><<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, fun);
}

template <typename F>
inline void parallel_for_gpu(cudaStream_t stream, size_t n_elements, F&& fun) {
	parallel_for_gpu(0, stream, n_elements, std::forward<F>(fun));
}

template <typename F>
inline void parallel_for_gpu(size_t n_elements, F&& fun) {
	parallel_for_gpu(nullptr, n_elements, std::forward<F>(fun));
}

#endif

// Optional sync to make NVTX ranges cover GPU time (may affect perf)
#ifndef TINYGS_NVTX_SYNC
#define TINYGS_NVTX_SYNC 0
#endif
inline void maybe_sync(cudaStream_t s = 0) {
#if TINYGS_NVTX_SYNC
  cudaStreamSynchronize(s);
#endif
}

template <typename T>
T from_string(const std::string& str);

}
