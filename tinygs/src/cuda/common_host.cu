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

/** @file   common_host.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#include <tinygs/cuda/common_device.cuh>
#include <tinygs/cuda/common_host.hpp>
// #include <tinygs/cuda/multi_stream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <unordered_map>

namespace tinygs {

static_assert(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2), "tiny-cuda-nn requires at least CUDA 10.2");

std::function<void(LogSeverity, const std::string&)> g_log_callback = [](LogSeverity severity, const std::string& msg) {
	switch (severity) {
		case LogSeverity::Warning: std::cerr << fmt::format("tiny-cuda-nn warning: {}\n", msg); break;
		case LogSeverity::Error: std::cerr << fmt::format("tiny-cuda-nn error: {}\n", msg); break;
		default: break;
	}

	if (verbose()) {
		switch (severity) {
			case LogSeverity::Debug: std::cerr << fmt::format("tiny-cuda-nn debug: {}\n", msg); break;
			case LogSeverity::Info: std::cerr << fmt::format("tiny-cuda-nn info: {}\n", msg); break;
			case LogSeverity::Success: std::cerr << fmt::format("tiny-cuda-nn success: {}\n", msg); break;
			default: break;
		}
	}
};

const std::function<void(LogSeverity, const std::string&)>& log_callback() { return g_log_callback; }
void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& cb) { g_log_callback = cb; }

bool g_verbose = false;
bool verbose() { return g_verbose; }
void set_verbose(bool verbose) { g_verbose = verbose; }

int cuda_runtime_version() {
	int version;
	CUDA_CHECK_THROW(cudaRuntimeGetVersion(&version));
	return version;
}

int cuda_device() {
	int device;
	CUDA_CHECK_THROW(cudaGetDevice(&device));
	return device;
}

void set_cuda_device(int device) { CUDA_CHECK_THROW(cudaSetDevice(device)); }

int cuda_device_count() {
	int device_count;
	CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
	return device_count;
}

bool cuda_supports_virtual_memory(int device) {
	int supports_vmm;
	CU_CHECK_THROW(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
	return supports_vmm != 0;
}

void check_features_supported() {
	int cuda_version = cuda_runtime_version();
	// 1. CUDA version > 11.0
	// 2. Device supports virtual memory
	if (cuda_version < 11000) {
			throw std::runtime_error{"CUDA version must be at least 11.0"};
	}

	int device = cuda_device();
	if (!cuda_supports_virtual_memory(device)) {
			throw std::runtime_error{fmt::format("Device {} does not support virtual memory management", device)};
	}
}

std::unordered_map<int, cudaDeviceProp>& cuda_device_properties() {
	static auto* cuda_device_props = new std::unordered_map<int, cudaDeviceProp>{};
	return *cuda_device_props;
}

const cudaDeviceProp& cuda_get_device_properties(int device) {
	if (cuda_device_properties().count(device) == 0) {
		auto& props = cuda_device_properties()[device];
		CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, device));
	}

	return cuda_device_properties().at(device);
}

std::string cuda_device_name(int device) { return cuda_get_device_properties(device).name; }

uint32_t cuda_compute_capability(int device) {
	const auto& props = cuda_get_device_properties(device);
	return props.major * 10 + props.minor;
}

uint32_t cuda_max_supported_compute_capability() {
	int cuda_version = cuda_runtime_version();
	if (cuda_version < 11000) {
		return 75;
	} else if (cuda_version < 11010) {
		return 80;
	} else if (cuda_version < 11080) {
		return 86;
	} else if (cuda_version < 12080) {
		return 90;
	} else {
		return 120;
	}
}

uint32_t cuda_supported_compute_capability(int device) {
	return std::min(cuda_compute_capability(device), cuda_max_supported_compute_capability());
}

size_t cuda_max_shmem(int device) { return cuda_get_device_properties(device).sharedMemPerBlockOptin; }

uint32_t cuda_max_registers(int device) { return (uint32_t)cuda_get_device_properties(device).regsPerBlock; }

size_t cuda_memory_granularity(int device) {
	size_t granularity;
	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = 0;
	CUresult granularity_result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	if (granularity_result == CUDA_ERROR_NOT_SUPPORTED) {
		return 1;
	}
	CU_CHECK_THROW(granularity_result);
	return granularity;
}

MemoryInfo cuda_memory_info() {
	MemoryInfo info;
	CUDA_CHECK_THROW(cudaMemGetInfo(&info.free, &info.total));
	info.used = info.total - info.free;
	return info;
}

std::string to_lower(std::string str) {
  std::transform(
      std::begin(str), std::end(str), std::begin(str),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return str;
}

std::string to_upper(std::string str) {
  std::transform(
      std::begin(str), std::end(str), std::begin(str),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return str;
}

} // namespace tcnn