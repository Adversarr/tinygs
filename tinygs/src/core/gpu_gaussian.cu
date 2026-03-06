#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cub/cub.cuh>
#include <memory>

#include "cuda/common_host.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/platform/runtime.hpp"
#include "utils/scope_timer.hpp"
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

struct GPUGaussian3d::Impl {
  std::shared_ptr<BackendRuntime> m_runtime;
  std::shared_ptr<BackendBuffer> m_means;
  std::shared_ptr<BackendBuffer> m_opacities;
  std::shared_ptr<BackendBuffer> m_rotations;
  std::shared_ptr<BackendBuffer> m_scales;
  std::shared_ptr<BackendBuffer> m_sh0;
  std::shared_ptr<BackendBuffer> m_sh1;
  std::shared_ptr<BackendBuffer> m_sh2;
  std::shared_ptr<BackendBuffer> m_sh3;
  size_t m_size = 0;
};

GPUGaussian3d::GPUGaussian3d(std::shared_ptr<BackendRuntime> runtime)
    : m_impl(std::make_unique<Impl>()) {
  m_impl->m_runtime = std::move(runtime);
}

GPUGaussian3d::~GPUGaussian3d() = default;

std::shared_ptr<BackendRuntime> GPUGaussian3d::runtime() const {
  return m_impl->m_runtime;
}

#define m_runtime m_impl->m_runtime
#define m_means m_impl->m_means
#define m_opacities m_impl->m_opacities
#define m_rotations m_impl->m_rotations
#define m_scales m_impl->m_scales
#define m_sh0 m_impl->m_sh0
#define m_sh1 m_impl->m_sh1
#define m_sh2 m_impl->m_sh2
#define m_sh3 m_impl->m_sh3
#define m_size m_impl->m_size

size_t GPUGaussian3d::size() const {
  return m_size;
}

DeviceSpan<const vec3> GPUGaussian3d::means() const {
  return DeviceSpan<const vec3>{m_means, 0, m_size};
}

DeviceSpan<const float> GPUGaussian3d::opacities() const {
  return DeviceSpan<const float>{m_opacities, 0, m_size};
}

DeviceSpan<const vec4> GPUGaussian3d::rotations() const {
  return DeviceSpan<const vec4>{m_rotations, 0, m_size};
}

DeviceSpan<const vec3> GPUGaussian3d::scales() const {
  return DeviceSpan<const vec3>{m_scales, 0, m_size};
}

DeviceSpan<vec3> GPUGaussian3d::means() {
  return DeviceSpan<vec3>{m_means, 0, m_size};
}

DeviceSpan<float> GPUGaussian3d::opacities() {
  return DeviceSpan<float>{m_opacities, 0, m_size};
}

DeviceSpan<vec4> GPUGaussian3d::rotations() {
  return DeviceSpan<vec4>{m_rotations, 0, m_size};
}

DeviceSpan<vec3> GPUGaussian3d::scales() {
  return DeviceSpan<vec3>{m_scales, 0, m_size};
}

DeviceSpan<const float> GPUGaussian3d::sh0() const {
  return DeviceSpan<const float>{m_sh0};
}

DeviceSpan<const float> GPUGaussian3d::sh1() const {
  return DeviceSpan<const float>{m_sh1};
}

DeviceSpan<const float> GPUGaussian3d::sh2() const {
  return DeviceSpan<const float>{m_sh2};
}

DeviceSpan<const float> GPUGaussian3d::sh3() const {
  return DeviceSpan<const float>{m_sh3};
}

DeviceSpan<float> GPUGaussian3d::sh0() {
  return DeviceSpan<float>{m_sh0};
}

DeviceSpan<float> GPUGaussian3d::sh1() {
  return DeviceSpan<float>{m_sh1};
}

DeviceSpan<float> GPUGaussian3d::sh2() {
  return DeviceSpan<float>{m_sh2};
}

DeviceSpan<float> GPUGaussian3d::sh3() {
  return DeviceSpan<float>{m_sh3};
}

__global__ void aos_to_soa_sh_kernel(
    const float3* __restrict__ src_aos,
    float* __restrict__ dst_soa,
    int N,
    int num_coeffs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * num_coeffs;
  if (idx >= total) return;

  int i = idx / num_coeffs;
  int k = idx % num_coeffs;

  float3 val = src_aos[i * num_coeffs + k];
  dst_soa[(k * 3 + 0) * N + i] = val.x;
  dst_soa[(k * 3 + 1) * N + i] = val.y;
  dst_soa[(k * 3 + 2) * N + i] = val.z;
}

__global__ void soa_to_aos_sh_kernel(
    const float* __restrict__ src_soa,
    float3* __restrict__ dst_aos,
    int N,
    int num_coeffs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * num_coeffs;
  if (idx >= total) return;

  int i = idx / num_coeffs;
  int k = idx % num_coeffs;

  float3 val;
  val.x = src_soa[(k * 3 + 0) * N + i];
  val.y = src_soa[(k * 3 + 1) * N + i];
  val.z = src_soa[(k * 3 + 2) * N + i];
  dst_aos[i * num_coeffs + k] = val;
}

static void upload_sh_aos_to_soa(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::vector<vec3>& host_aos,
    std::shared_ptr<BackendBuffer>& gpu_soa,
    int N, int num_coeffs) {
  if (N == 0 || num_coeffs == 0) {
    gpu_soa.reset();
    return;
  }
  auto temp_aos = create_device_buffer_for<float3>(runtime, N * num_coeffs, "sh_temp_aos");
  tinygs::copy_from_host(runtime, queue, temp_aos,
      reinterpret_cast<const float3*>(host_aos.data()), N * num_coeffs);

  gpu_soa = create_device_buffer_for<float>(runtime, num_coeffs * 3 * N, "sh_soa");
  int total = N * num_coeffs;
  int blocks = (total + 255) / 256;
  aos_to_soa_sh_kernel<<<blocks, 256>>>(
      buffer_data<float3>(temp_aos),
      buffer_data<float>(gpu_soa),
      N, num_coeffs);
  CUDA_CHECK_THROW(cudaGetLastError());
}

static void download_sh_soa_to_aos(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& gpu_soa,
    std::vector<vec3>& host_aos,
    int N, int num_coeffs) {
  if (N == 0 || num_coeffs == 0) {
    host_aos.clear();
    return;
  }
  auto temp_aos = create_device_buffer_for<float3>(runtime, N * num_coeffs, "sh_temp_aos");
  int total = N * num_coeffs;
  int blocks = (total + 255) / 256;
  soa_to_aos_sh_kernel<<<blocks, 256>>>(
      buffer_data<float>(gpu_soa),
      buffer_data<float3>(temp_aos),
      N, num_coeffs);
  CUDA_CHECK_THROW(cudaGetLastError());

  host_aos.resize(N * num_coeffs);
  tinygs::copy_to_host(runtime, queue, temp_aos,
      reinterpret_cast<float3*>(host_aos.data()), N * num_coeffs);
}

float* GPUGaussian3d::sh_degree_data(int degree) {
  switch (degree) {
    case 0: return buffer_data<float>(m_sh0);
    case 1: return buffer_data<float>(m_sh1);
    case 2: return buffer_data<float>(m_sh2);
    case 3: return buffer_data<float>(m_sh3);
    default: return nullptr;
  }
}

const float* GPUGaussian3d::sh_degree_data(int degree) const {
  switch (degree) {
    case 0: return buffer_data_const<float>(m_sh0);
    case 1: return buffer_data_const<float>(m_sh1);
    case 2: return buffer_data_const<float>(m_sh2);
    case 3: return buffer_data_const<float>(m_sh3);
    default: return nullptr;
  }
}

void GPUGaussian3d::copy_from_host(const Gaussian3d& gaussians, const std::shared_ptr<BackendQueue>& queue) {
  NVTX3_FUNC_RANGE();
  const int N = static_cast<int>(gaussians.means.size());
  m_size = N;

  m_means = create_device_buffer_for<vec3>(m_runtime, N, "means");
  m_opacities = create_device_buffer_for<float>(m_runtime, N, "opacities");
  m_rotations = create_device_buffer_for<vec4>(m_runtime, N, "rotations");
  m_scales = create_device_buffer_for<vec3>(m_runtime, N, "scales");

  tinygs::copy_from_host(m_runtime, queue, m_means, gaussians.means.data(), N);
  tinygs::copy_from_host(m_runtime, queue, m_opacities, gaussians.opacities.data(), N);
  tinygs::copy_from_host(m_runtime, queue, m_rotations, gaussians.rotations.data(), N);
  tinygs::copy_from_host(m_runtime, queue, m_scales, gaussians.scales.data(), N);

  upload_sh_aos_to_soa(m_runtime, queue, gaussians.sh0, m_sh0, N, 1);
  upload_sh_aos_to_soa(m_runtime, queue, gaussians.sh1, m_sh1, N, 3);
  upload_sh_aos_to_soa(m_runtime, queue, gaussians.sh2, m_sh2, N, 5);
  upload_sh_aos_to_soa(m_runtime, queue, gaussians.sh3, m_sh3, N, 7);
}

void GPUGaussian3d::copy_to_host(Gaussian3d& gaussians, const std::shared_ptr<BackendQueue>& queue) {
  NVTX3_FUNC_RANGE();
  const int N = static_cast<int>(m_size);

  gaussians.means.resize(N);
  gaussians.opacities.resize(N);
  gaussians.rotations.resize(N);
  gaussians.scales.resize(N);

  tinygs::copy_to_host(m_runtime, queue, m_means, gaussians.means.data(), N);
  tinygs::copy_to_host(m_runtime, queue, m_opacities, gaussians.opacities.data(), N);
  tinygs::copy_to_host(m_runtime, queue, m_rotations, gaussians.rotations.data(), N);
  tinygs::copy_to_host(m_runtime, queue, m_scales, gaussians.scales.data(), N);

  download_sh_soa_to_aos(m_runtime, queue, m_sh0, gaussians.sh0, N, 1);
  download_sh_soa_to_aos(m_runtime, queue, m_sh1, gaussians.sh1, N, 3);
  download_sh_soa_to_aos(m_runtime, queue, m_sh2, gaussians.sh2, N, 5);
  download_sh_soa_to_aos(m_runtime, queue, m_sh3, gaussians.sh3, N, 7);
}

void GPUGaussian3d::memset_async(char value, BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  if (m_means && m_means->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_means), value, m_means->size_bytes(), cuda_stream));
  if (m_opacities && m_opacities->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_opacities), value, m_opacities->size_bytes(), cuda_stream));
  if (m_rotations && m_rotations->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_rotations), value, m_rotations->size_bytes(), cuda_stream));
  if (m_scales && m_scales->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_scales), value, m_scales->size_bytes(), cuda_stream));
  if (m_sh0 && m_sh0->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_sh0), value, m_sh0->size_bytes(), cuda_stream));
  if (m_sh1 && m_sh1->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_sh1), value, m_sh1->size_bytes(), cuda_stream));
  if (m_sh2 && m_sh2->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_sh2), value, m_sh2->size_bytes(), cuda_stream));
  if (m_sh3 && m_sh3->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemsetAsync(buffer_data<void>(m_sh3), value, m_sh3->size_bytes(), cuda_stream));
}

void GPUGaussian3d::memset(char value) {
  if (m_means && m_means->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_means), value, m_means->size_bytes()));
  if (m_opacities && m_opacities->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_opacities), value, m_opacities->size_bytes()));
  if (m_rotations && m_rotations->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_rotations), value, m_rotations->size_bytes()));
  if (m_scales && m_scales->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_scales), value, m_scales->size_bytes()));
  if (m_sh0 && m_sh0->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_sh0), value, m_sh0->size_bytes()));
  if (m_sh1 && m_sh1->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_sh1), value, m_sh1->size_bytes()));
  if (m_sh2 && m_sh2->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_sh2), value, m_sh2->size_bytes()));
  if (m_sh3 && m_sh3->size_bytes() > 0)
    CUDA_CHECK_THROW(cudaMemset(buffer_data<void>(m_sh3), value, m_sh3->size_bytes()));
}

template<typename IndexType>
__global__ void copy_gaussian_base_items(
    const vec3* __restrict__ src_means,
    vec3* __restrict__ dst_means,
    const float* __restrict__ src_opacities,
    float* __restrict__ dst_opacities,
    const vec4* __restrict__ src_rotations,
    vec4* __restrict__ dst_rotations,
    const vec3* __restrict__ src_scales,
    vec3* __restrict__ dst_scales,
    const IndexType* __restrict__ mapping,
    int num_items) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items) return;

  IndexType src_idx = mapping[idx];
  dst_means[idx] = src_means[src_idx];
  dst_opacities[idx] = src_opacities[src_idx];
  dst_rotations[idx] = src_rotations[src_idx];
  dst_scales[idx] = src_scales[src_idx];
}

template<typename IndexType>
__global__ void gather_soa_sh_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const IndexType* __restrict__ mapping,
    int new_N,
    int old_N,
    int num_coeffs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = new_N * num_coeffs * 3;
  if (idx >= total) return;

  int i = idx % new_N;
  int kc = idx / new_N;
  IndexType src_idx = mapping[i];

  dst[kc * new_N + i] = src[kc * old_N + src_idx];
}

__global__ void relayout_soa_sh_append_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int old_N,
    int new_N,
    int num_coeffs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_channels = num_coeffs * 3;
  int total = old_N * num_channels;
  if (idx >= total) return;

  int i = idx % old_N;
  int kc = idx / old_N;
  dst[kc * new_N + i] = src[kc * old_N + i];
}

template<typename IndexType>
static void gather_soa_sh(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendBuffer>& src,
    std::shared_ptr<BackendBuffer>& dst,
    const IndexType* mapping,
    int new_N, int old_N, int num_coeffs,
    BackendStream stream = nullptr) {
  if (num_coeffs == 0 || new_N == 0) {
    dst = create_device_buffer_for<float>(runtime, num_coeffs * 3 * new_N, "sh_gather");
    return;
  }
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  dst = create_device_buffer_for<float>(runtime, num_coeffs * 3 * new_N, "sh_gather");
  int total = new_N * num_coeffs * 3;
  int blocks = (total + 255) / 256;
  gather_soa_sh_kernel<IndexType><<<blocks, 256, 0, cuda_stream>>>(
      buffer_data<float>(src),
      buffer_data<float>(dst),
      mapping,
      new_N, old_N, num_coeffs);
}

void GPUGaussian3d::remove(char* kept_flag, int num_kept, BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  size_t original_size = size();
  
  auto mapping = create_device_buffer_for<int>(m_runtime, original_size, "remove_mapping");
  
  int* d_mapping = buffer_data<int>(mapping);
  thrust::copy_if(
      thrust::cuda::par.on(cuda_stream),
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(original_size),
      d_mapping,
      [kept_flag] __device__(int orig) { return static_cast<bool>(kept_flag[orig]); });

  auto new_means = create_device_buffer_for<vec3>(m_runtime, num_kept, "means");
  auto new_opacities = create_device_buffer_for<float>(m_runtime, num_kept, "opacities");
  auto new_rotations = create_device_buffer_for<vec4>(m_runtime, num_kept, "rotations");
  auto new_scales = create_device_buffer_for<vec3>(m_runtime, num_kept, "scales");

  const int grid = (num_kept + 255) / 256;
  copy_gaussian_base_items<int><<<grid, 256, 0, cuda_stream>>>(
      buffer_data<vec3>(m_means),
      buffer_data<vec3>(new_means),
      buffer_data<float>(m_opacities),
      buffer_data<float>(new_opacities),
      buffer_data<vec4>(m_rotations),
      buffer_data<vec4>(new_rotations),
      buffer_data<vec3>(m_scales),
      buffer_data<vec3>(new_scales),
      d_mapping,
      num_kept);

  std::shared_ptr<BackendBuffer> sh0_new, sh1_new, sh2_new, sh3_new;
  gather_soa_sh(m_runtime, m_sh0, sh0_new, d_mapping, num_kept, (int)original_size, 1, stream);
  gather_soa_sh(m_runtime, m_sh1, sh1_new, d_mapping, num_kept, (int)original_size, 3, stream);
  gather_soa_sh(m_runtime, m_sh2, sh2_new, d_mapping, num_kept, (int)original_size, 5, stream);
  gather_soa_sh(m_runtime, m_sh3, sh3_new, d_mapping, num_kept, (int)original_size, 7, stream);

  m_means = std::move(new_means);
  m_opacities = std::move(new_opacities);
  m_rotations = std::move(new_rotations);
  m_scales = std::move(new_scales);
  m_sh0 = std::move(sh0_new);
  m_sh1 = std::move(sh1_new);
  m_sh2 = std::move(sh2_new);
  m_sh3 = std::move(sh3_new);
  m_size = num_kept;
}

void GPUGaussian3d::append(int num_dup, const std::shared_ptr<BackendQueue>& queue) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(queue->native_handle());
  assert(num_dup > 0);
  const size_t target_size = this->size() + static_cast<size_t>(num_dup);
  const size_t old_size = this->size();

  auto new_means = create_device_buffer_for<vec3>(m_runtime, target_size, "means");
  auto new_opacities = create_device_buffer_for<float>(m_runtime, target_size, "opacities");
  auto new_rotations = create_device_buffer_for<vec4>(m_runtime, target_size, "rotations");
  auto new_scales = create_device_buffer_for<vec3>(m_runtime, target_size, "scales");

  fill_buffer_zero(m_runtime, queue, new_means);
  fill_buffer_zero(m_runtime, queue, new_opacities);
  fill_buffer_zero(m_runtime, queue, new_rotations);
  fill_buffer_zero(m_runtime, queue, new_scales);

  if (m_means && m_means->size_bytes() > 0) {
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(new_means),
        buffer_data<void>(m_means),
        old_size * sizeof(vec3),
        cudaMemcpyDeviceToDevice, cuda_stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(new_opacities),
        buffer_data<void>(m_opacities),
        old_size * sizeof(float),
        cudaMemcpyDeviceToDevice, cuda_stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(new_rotations),
        buffer_data<void>(m_rotations),
        old_size * sizeof(vec4),
        cudaMemcpyDeviceToDevice, cuda_stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(new_scales),
        buffer_data<void>(m_scales),
        old_size * sizeof(vec3),
        cudaMemcpyDeviceToDevice, cuda_stream));
  }

  auto resize_soa_sh = [&](std::shared_ptr<BackendBuffer>& buf, int num_coeffs) {
    if (num_coeffs == 0) return;
    int new_total = num_coeffs * 3 * static_cast<int>(target_size);

    auto new_buf = create_device_buffer_for<float>(m_runtime, new_total, "sh_resize");
    fill_buffer_zero(m_runtime, queue, new_buf);

    if (old_size == 0 || !buf || buf->size_bytes() == 0) {
      buf = std::move(new_buf);
      return;
    }

    int total_elems = static_cast<int>(old_size) * num_coeffs * 3;
    int blocks = (total_elems + 255) / 256;
    relayout_soa_sh_append_kernel<<<blocks, 256, 0, cuda_stream>>>(
        buffer_data<float>(buf),
        buffer_data<float>(new_buf),
        static_cast<int>(old_size),
        static_cast<int>(target_size),
        num_coeffs);
    CUDA_CHECK_THROW(cudaGetLastError());
    buf = std::move(new_buf);
  };

  resize_soa_sh(m_sh0, 1);
  resize_soa_sh(m_sh1, 3);
  resize_soa_sh(m_sh2, 5);
  resize_soa_sh(m_sh3, 7);

  m_means = std::move(new_means);
  m_opacities = std::move(new_opacities);
  m_rotations = std::move(new_rotations);
  m_scales = std::move(new_scales);
  m_size = target_size;
}

static void copy_sh_async(
    std::shared_ptr<BackendBuffer>& dst,
    const std::shared_ptr<BackendBuffer>& src,
    const std::shared_ptr<BackendRuntime>& runtime,
    BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  if (!src || src->size_bytes() == 0) {
    dst.reset();
    return;
  }
  dst = create_device_buffer(runtime, src->size_bytes(), "sh_copy");
  CUDA_CHECK_THROW(cudaMemcpyAsync(
      buffer_data<void>(dst),
      buffer_data<void>(src),
      src->size_bytes(),
      cudaMemcpyDeviceToDevice,
      cuda_stream));
}

static void copy_sh_sync(
    std::shared_ptr<BackendBuffer>& dst,
    const std::shared_ptr<BackendBuffer>& src,
    const std::shared_ptr<BackendRuntime>& runtime) {
  if (!src || src->size_bytes() == 0) {
    dst.reset();
    return;
  }
  dst = create_device_buffer(runtime, src->size_bytes(), "sh_copy");
  CUDA_CHECK_THROW(cudaMemcpy(
      buffer_data<void>(dst),
      buffer_data<void>(src),
      src->size_bytes(),
      cudaMemcpyDeviceToDevice));
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone_async(BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  auto gaussians = std::make_unique<GPUGaussian3d>(m_runtime);
  gaussians->m_current_sh_degree = m_current_sh_degree;
  gaussians->m_scene_scale = m_scene_scale;
  gaussians->m_size = m_size;

  if (m_means && m_means->size_bytes() > 0) {
    gaussians->m_means = create_device_buffer(m_runtime, m_means->size_bytes(), "means");
    gaussians->m_opacities = create_device_buffer(m_runtime, m_opacities->size_bytes(), "opacities");
    gaussians->m_rotations = create_device_buffer(m_runtime, m_rotations->size_bytes(), "rotations");
    gaussians->m_scales = create_device_buffer(m_runtime, m_scales->size_bytes(), "scales");

    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(gaussians->m_means),
        buffer_data<void>(m_means),
        m_means->size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(gaussians->m_opacities),
        buffer_data<void>(m_opacities),
        m_opacities->size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(gaussians->m_rotations),
        buffer_data<void>(m_rotations),
        m_rotations->size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream));
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        buffer_data<void>(gaussians->m_scales),
        buffer_data<void>(m_scales),
        m_scales->size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream));
  }

  copy_sh_async(gaussians->m_sh0, m_sh0, m_runtime, stream);
  copy_sh_async(gaussians->m_sh1, m_sh1, m_runtime, stream);
  copy_sh_async(gaussians->m_sh2, m_sh2, m_runtime, stream);
  copy_sh_async(gaussians->m_sh3, m_sh3, m_runtime, stream);

  return gaussians;
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone() {
  auto gaussians = std::make_unique<GPUGaussian3d>(m_runtime);
  gaussians->m_current_sh_degree = m_current_sh_degree;
  gaussians->m_scene_scale = m_scene_scale;
  gaussians->m_size = m_size;

  if (m_means && m_means->size_bytes() > 0) {
    gaussians->m_means = create_device_buffer(m_runtime, m_means->size_bytes(), "means");
    gaussians->m_opacities = create_device_buffer(m_runtime, m_opacities->size_bytes(), "opacities");
    gaussians->m_rotations = create_device_buffer(m_runtime, m_rotations->size_bytes(), "rotations");
    gaussians->m_scales = create_device_buffer(m_runtime, m_scales->size_bytes(), "scales");

    CUDA_CHECK_THROW(cudaMemcpy(
        buffer_data<void>(gaussians->m_means),
        buffer_data<void>(m_means),
        m_means->size_bytes(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(
        buffer_data<void>(gaussians->m_opacities),
        buffer_data<void>(m_opacities),
        m_opacities->size_bytes(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(
        buffer_data<void>(gaussians->m_rotations),
        buffer_data<void>(m_rotations),
        m_rotations->size_bytes(), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(
        buffer_data<void>(gaussians->m_scales),
        buffer_data<void>(m_scales),
        m_scales->size_bytes(), cudaMemcpyDeviceToDevice));
  }

  copy_sh_sync(gaussians->m_sh0, m_sh0, m_runtime);
  copy_sh_sync(gaussians->m_sh1, m_sh1, m_runtime);
  copy_sh_sync(gaussians->m_sh2, m_sh2, m_runtime);
  copy_sh_sync(gaussians->m_sh3, m_sh3, m_runtime);

  return gaussians;
}

void GPUGaussian3d::reorder(uint* indices, BackendStream stream) {
  NVTX3_FUNC_RANGE();
  const int N = static_cast<int>(size());
  const cudaStream_t cuda_stream = to_cuda_stream(stream);

  auto new_means = create_device_buffer_for<vec3>(m_runtime, N, "means");
  auto new_opacities = create_device_buffer_for<float>(m_runtime, N, "opacities");
  auto new_rotations = create_device_buffer_for<vec4>(m_runtime, N, "rotations");
  auto new_scales = create_device_buffer_for<vec3>(m_runtime, N, "scales");

  const int grid = (N + 255) / 256;
  copy_gaussian_base_items<uint><<<grid, 256, 0, cuda_stream>>>(
      buffer_data<vec3>(m_means),
      buffer_data<vec3>(new_means),
      buffer_data<float>(m_opacities),
      buffer_data<float>(new_opacities),
      buffer_data<vec4>(m_rotations),
      buffer_data<vec4>(new_rotations),
      buffer_data<vec3>(m_scales),
      buffer_data<vec3>(new_scales),
      indices, N);

  std::shared_ptr<BackendBuffer> sh0_new, sh1_new, sh2_new, sh3_new;
  gather_soa_sh(m_runtime, m_sh0, sh0_new, indices, N, N, 1, stream);
  gather_soa_sh(m_runtime, m_sh1, sh1_new, indices, N, N, 3, stream);
  gather_soa_sh(m_runtime, m_sh2, sh2_new, indices, N, N, 5, stream);
  gather_soa_sh(m_runtime, m_sh3, sh3_new, indices, N, N, 7, stream);

  m_means = std::move(new_means);
  m_opacities = std::move(new_opacities);
  m_rotations = std::move(new_rotations);
  m_scales = std::move(new_scales);
  m_sh0 = std::move(sh0_new);
  m_sh1 = std::move(sh1_new);
  m_sh2 = std::move(sh2_new);
  m_sh3 = std::move(sh3_new);
}

#undef m_runtime
#undef m_means
#undef m_opacities
#undef m_rotations
#undef m_scales
#undef m_sh0
#undef m_sh1
#undef m_sh2
#undef m_sh3
#undef m_size

}  // namespace tinygs
