#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "tinygs/platform/runtime.hpp"
#include "tinygs/platform/buffer_utils.hpp"

namespace tinygs::optim_detail {

template <typename IndexType>
__global__ static void gather_soa_optim_kernel(const float* __restrict__ src,
                                                float* __restrict__ dst,
                                                const IndexType* __restrict__ mapping,
                                                int num_items,
                                                int num_channels,
                                                int src_stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_items * num_channels) {
    return;
  }
  int item_idx = tid % num_items;
  int channel = tid / num_items;
  dst[channel * num_items + item_idx] = src[channel * src_stride + mapping[item_idx]];
}

template <typename T>
__global__ static void duplicate_optim_buffer_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int* __restrict__ src_indices,
    const int* __restrict__ dst_indices,
    int num_items) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_items) {
    return;
  }
  dst[dst_indices[tid]] = src[src_indices[tid]];
}

__global__ static void duplicate_soa_optim_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ src_indices,
    const int* __restrict__ dst_indices,
    int num_items,
    int num_channels,
    int stride) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_items * num_channels) {
    return;
  }
  const int item_idx = tid % num_items;
  const int channel = tid / num_items;
  dst[channel * stride + dst_indices[item_idx]] = src[channel * stride + src_indices[item_idx]];
}

inline void gather_soa_optim_buffers(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& src_first,
    const std::shared_ptr<BackendBuffer>& src_second,
    std::shared_ptr<BackendBuffer>& dst_first,
    std::shared_ptr<BackendBuffer>& dst_second,
    const unsigned int* mapping,
    int num_items,
    int num_channels,
    int src_stride,
    int block_size) {
  int total = num_items * num_channels;
  dst_first = create_device_buffer_for<float>(runtime, total, "optim_first");
  dst_second = create_device_buffer_for<float>(runtime, total, "optim_second");
  fill_buffer_zero_async(runtime, queue, dst_first);
  fill_buffer_zero_async(runtime, queue, dst_second);
  if (total == 0) {
    return;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(queue->native_handle());
  const int grid = (total + block_size - 1) / block_size;
  gather_soa_optim_kernel<unsigned int><<<grid, block_size, 0, stream>>>(
      buffer_data<float>(src_first),
      buffer_data<float>(dst_first),
      mapping,
      num_items,
      num_channels,
      src_stride);
  gather_soa_optim_kernel<unsigned int><<<grid, block_size, 0, stream>>>(
      buffer_data<float>(src_second),
      buffer_data<float>(dst_second),
      mapping,
      num_items,
      num_channels,
      src_stride);
}

inline void relayout_soa_optim(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    std::shared_ptr<BackendBuffer>& buf,
    int old_n,
    int new_n,
    int num_channels,
    int block_size) {
  if (old_n == 0 || new_n == 0) {
    buf = create_device_buffer_for<float>(runtime, new_n * num_channels, "optim_relayout");
    fill_buffer_zero_async(runtime, queue, buf);
    return;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(queue->native_handle());
  auto identity = create_device_buffer_for<unsigned int>(runtime, old_n, "identity_mapping");
  thrust::sequence(thrust::cuda::par.on(stream),
                   buffer_data<unsigned int>(identity),
                   buffer_data<unsigned int>(identity) + old_n);

  auto new_buf = create_device_buffer_for<float>(runtime, new_n * num_channels, "optim_relayout");
  fill_buffer_zero_async(runtime, queue, new_buf);
  
  int total = old_n * num_channels;
  const int grid = (total + block_size - 1) / block_size;
  gather_soa_optim_kernel<unsigned int><<<grid, block_size, 0, stream>>>(
      buffer_data<float>(buf),
      buffer_data<float>(new_buf),
      buffer_data<unsigned int>(identity),
      old_n,
      num_channels,
      old_n);
  buf = std::move(new_buf);
}

template <typename T>
inline void duplicate_optim_buffer(
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buf,
    const int* src_indices,
    const int* dst_indices,
    int num_items,
    int block_size) {
  if (num_items == 0) {
    return;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(queue->native_handle());
  const int grid = (num_items + block_size - 1) / block_size;
  duplicate_optim_buffer_kernel<T><<<grid, block_size, 0, stream>>>(
      buffer_data<T>(buf),
      buffer_data<T>(buf),
      src_indices,
      dst_indices,
      num_items);
}

inline void duplicate_soa_optim_buffer(
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& buf,
    const int* src_indices,
    const int* dst_indices,
    int num_items,
    int num_channels,
    int stride,
    int block_size) {
  const int total = num_items * num_channels;
  if (total == 0) {
    return;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(queue->native_handle());
  const int grid = (total + block_size - 1) / block_size;
  duplicate_soa_optim_kernel<<<grid, block_size, 0, stream>>>(
      buffer_data<float>(buf),
      buffer_data<float>(buf),
      src_indices,
      dst_indices,
      num_items,
      num_channels,
      stride);
}

}  // namespace tinygs::optim_detail
