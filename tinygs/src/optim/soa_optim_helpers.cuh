#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

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

inline void gather_soa_optim_buffers(const thrust::device_vector<float>& src_first,
                                     const thrust::device_vector<float>& src_second,
                                     thrust::device_vector<float>& dst_first,
                                     thrust::device_vector<float>& dst_second,
                                     const unsigned int* mapping,
                                     int num_items,
                                     int num_channels,
                                     int src_stride,
                                     int block_size,
                                     cudaStream_t stream = nullptr) {
  int total = num_items * num_channels;
  dst_first.resize(total, 0.f);
  dst_second.resize(total, 0.f);
  if (total == 0) {
    return;
  }
  const int grid = (total + block_size - 1) / block_size;
  gather_soa_optim_kernel<unsigned int><<<grid, block_size, 0, stream>>>(
      thrust::raw_pointer_cast(src_first.data()),
      thrust::raw_pointer_cast(dst_first.data()),
      mapping,
      num_items,
      num_channels,
      src_stride);
  gather_soa_optim_kernel<unsigned int><<<grid, block_size, 0, stream>>>(
      thrust::raw_pointer_cast(src_second.data()),
      thrust::raw_pointer_cast(dst_second.data()),
      mapping,
      num_items,
      num_channels,
      src_stride);
}

inline void relayout_soa_optim(thrust::device_vector<float>& buf,
                               int old_n,
                               int new_n,
                               int num_channels,
                               int block_size,
                               cudaStream_t stream = nullptr) {
  if (old_n == 0 || new_n == 0) {
    buf.assign(new_n * num_channels, 0.f);
    return;
  }

  thrust::device_vector<unsigned int> identity(old_n);
  thrust::sequence(thrust::cuda::par.on(stream), identity.begin(), identity.end());

  thrust::device_vector<float> new_buf(new_n * num_channels, 0.f);
  int total = old_n * num_channels;
  const int grid = (total + block_size - 1) / block_size;
  gather_soa_optim_kernel<unsigned int><<<grid, block_size, 0, stream>>>(
      thrust::raw_pointer_cast(buf.data()),
      thrust::raw_pointer_cast(new_buf.data()),
      thrust::raw_pointer_cast(identity.data()),
      old_n,
      num_channels,
      old_n);
  buf = std::move(new_buf);
}

}  // namespace tinygs::optim_detail
