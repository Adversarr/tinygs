#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cub/cub.cuh>
#include <memory>

#include "cuda/common_host.hpp"
#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/cuda/gpu_memory.hpp"
#include "utils/scope_timer.hpp"
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

struct GPUGaussian3d::Impl {
  thrust::device_vector<vec3> m_means;
  thrust::device_vector<float> m_opacities;
  thrust::device_vector<vec4> m_rotations;
  thrust::device_vector<vec3> m_scales;
  thrust::device_vector<float> m_sh0;
  thrust::device_vector<float> m_sh1;
  thrust::device_vector<float> m_sh2;
  thrust::device_vector<float> m_sh3;
};

GPUGaussian3d::GPUGaussian3d() : m_impl(std::make_unique<Impl>()) {}

GPUGaussian3d::~GPUGaussian3d() = default;

#define m_means m_impl->m_means
#define m_opacities m_impl->m_opacities
#define m_rotations m_impl->m_rotations
#define m_scales m_impl->m_scales
#define m_sh0 m_impl->m_sh0
#define m_sh1 m_impl->m_sh1
#define m_sh2 m_impl->m_sh2
#define m_sh3 m_impl->m_sh3

size_t GPUGaussian3d::size() const {
  return m_means.size();
}

DeviceSpan<const vec3> GPUGaussian3d::means() const {
  return DeviceSpan<const vec3>{thrust::raw_pointer_cast(m_means.data()), m_means.size()};
}

DeviceSpan<const float> GPUGaussian3d::opacities() const {
  return DeviceSpan<const float>{thrust::raw_pointer_cast(m_opacities.data()), m_opacities.size()};
}

DeviceSpan<const vec4> GPUGaussian3d::rotations() const {
  return DeviceSpan<const vec4>{thrust::raw_pointer_cast(m_rotations.data()), m_rotations.size()};
}

DeviceSpan<const vec3> GPUGaussian3d::scales() const {
  return DeviceSpan<const vec3>{thrust::raw_pointer_cast(m_scales.data()), m_scales.size()};
}

DeviceSpan<vec3> GPUGaussian3d::means() {
  return DeviceSpan<vec3>{thrust::raw_pointer_cast(m_means.data()), m_means.size()};
}

DeviceSpan<float> GPUGaussian3d::opacities() {
  return DeviceSpan<float>{thrust::raw_pointer_cast(m_opacities.data()), m_opacities.size()};
}

DeviceSpan<vec4> GPUGaussian3d::rotations() {
  return DeviceSpan<vec4>{thrust::raw_pointer_cast(m_rotations.data()), m_rotations.size()};
}

DeviceSpan<vec3> GPUGaussian3d::scales() {
  return DeviceSpan<vec3>{thrust::raw_pointer_cast(m_scales.data()), m_scales.size()};
}

DeviceSpan<const float> GPUGaussian3d::sh0() const {
  return DeviceSpan<const float>{thrust::raw_pointer_cast(m_sh0.data()), m_sh0.size()};
}

DeviceSpan<const float> GPUGaussian3d::sh1() const {
  return DeviceSpan<const float>{thrust::raw_pointer_cast(m_sh1.data()), m_sh1.size()};
}

DeviceSpan<const float> GPUGaussian3d::sh2() const {
  return DeviceSpan<const float>{thrust::raw_pointer_cast(m_sh2.data()), m_sh2.size()};
}

DeviceSpan<const float> GPUGaussian3d::sh3() const {
  return DeviceSpan<const float>{thrust::raw_pointer_cast(m_sh3.data()), m_sh3.size()};
}

DeviceSpan<float> GPUGaussian3d::sh0() {
  return DeviceSpan<float>{thrust::raw_pointer_cast(m_sh0.data()), m_sh0.size()};
}

DeviceSpan<float> GPUGaussian3d::sh1() {
  return DeviceSpan<float>{thrust::raw_pointer_cast(m_sh1.data()), m_sh1.size()};
}

DeviceSpan<float> GPUGaussian3d::sh2() {
  return DeviceSpan<float>{thrust::raw_pointer_cast(m_sh2.data()), m_sh2.size()};
}

DeviceSpan<float> GPUGaussian3d::sh3() {
  return DeviceSpan<float>{thrust::raw_pointer_cast(m_sh3.data()), m_sh3.size()};
}

// ============================================================================
// AoS <-> SoA conversion kernels for SH coefficients
// ============================================================================

/// @brief Convert SH coefficients from AoS (vec3 per coeff, contiguous per Gaussian)
///        to channel-first SoA layout: [c0_R_all, c0_G_all, c0_B_all, c1_R_all, ...]
///
/// AoS input layout (CPU):  For N gaussians, C coefficients:
///   [G0_c0_RGB, G0_c1_RGB, ..., G0_c{C-1}_RGB, G1_c0_RGB, ...]
///
/// SoA output layout (GPU): For coefficient k, channel c, Gaussian i:
///   index = (k * 3 + c) * N + i
__global__ void aos_to_soa_sh_kernel(
    const float3* __restrict__ src_aos,
    float* __restrict__ dst_soa,
    int N,
    int num_coeffs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * num_coeffs;
  if (idx >= total) return;

  int i = idx / num_coeffs;  // Gaussian index
  int k = idx % num_coeffs;  // Coefficient index

  float3 val = src_aos[i * num_coeffs + k];
  dst_soa[(k * 3 + 0) * N + i] = val.x;  // R channel
  dst_soa[(k * 3 + 1) * N + i] = val.y;  // G channel
  dst_soa[(k * 3 + 2) * N + i] = val.z;  // B channel
}

/// @brief Convert SH coefficients from channel-first SoA layout back to AoS (vec3).
__global__ void soa_to_aos_sh_kernel(
    const float* __restrict__ src_soa,
    float3* __restrict__ dst_aos,
    int N,
    int num_coeffs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * num_coeffs;
  if (idx >= total) return;

  int i = idx / num_coeffs;  // Gaussian index
  int k = idx % num_coeffs;  // Coefficient index

  float3 val;
  val.x = src_soa[(k * 3 + 0) * N + i];
  val.y = src_soa[(k * 3 + 1) * N + i];
  val.z = src_soa[(k * 3 + 2) * N + i];
  dst_aos[i * num_coeffs + k] = val;
}

/// @brief Upload SH data from CPU AoS (vec3) to GPU SoA (float) layout.
/// @param host_aos  Host-side AoS data: N * num_coeffs vec3 elements.
/// @param gpu_soa   Device-side SoA buffer: num_coeffs * 3 * N floats.
/// @param N         Number of Gaussians.
/// @param num_coeffs Number of coefficients per Gaussian for this degree.
static void upload_sh_aos_to_soa(
    const std::vector<vec3>& host_aos,
    thrust::device_vector<float>& gpu_soa,
    int N, int num_coeffs) {
  if (N == 0 || num_coeffs == 0) {
    gpu_soa.clear();
    return;
  }
  // Upload AoS to temporary GPU buffer
  thrust::device_vector<float3> temp_aos(N * num_coeffs);
  thrust::copy(
      reinterpret_cast<const float3*>(host_aos.data()),
      reinterpret_cast<const float3*>(host_aos.data()) + N * num_coeffs,
      temp_aos.begin());

  // Resize SoA output and convert on device
  gpu_soa.resize(num_coeffs * 3 * N);
  int total = N * num_coeffs;
  int blocks = (total + 255) / 256;
  aos_to_soa_sh_kernel<<<blocks, 256>>>(
      thrust::raw_pointer_cast(temp_aos.data()),
      thrust::raw_pointer_cast(gpu_soa.data()),
      N, num_coeffs);
  CUDA_CHECK_THROW(cudaGetLastError());
}

/// @brief Download SH data from GPU SoA (float) to CPU AoS (vec3) layout.
static void download_sh_soa_to_aos(
    const thrust::device_vector<float>& gpu_soa,
    std::vector<vec3>& host_aos,
    int N, int num_coeffs) {
  if (N == 0 || num_coeffs == 0) {
    host_aos.clear();
    return;
  }
  // Convert SoA -> AoS on device
  thrust::device_vector<float3> temp_aos(N * num_coeffs);
  int total = N * num_coeffs;
  int blocks = (total + 255) / 256;
  soa_to_aos_sh_kernel<<<blocks, 256>>>(
      thrust::raw_pointer_cast(gpu_soa.data()),
      thrust::raw_pointer_cast(temp_aos.data()),
      N, num_coeffs);
  CUDA_CHECK_THROW(cudaGetLastError());

  // Download to host
  host_aos.resize(N * num_coeffs);
  thrust::copy(temp_aos.begin(), temp_aos.end(),
               reinterpret_cast<float3*>(host_aos.data()));
}

// ============================================================================
// GPUGaussian3d: sh_degree_data() accessor
// ============================================================================

float* GPUGaussian3d::sh_degree_data(int degree) {
  switch (degree) {
    case 0: return thrust::raw_pointer_cast(m_sh0.data());
    case 1: return thrust::raw_pointer_cast(m_sh1.data());
    case 2: return thrust::raw_pointer_cast(m_sh2.data());
    case 3: return thrust::raw_pointer_cast(m_sh3.data());
    default: return nullptr;
  }
}

const float* GPUGaussian3d::sh_degree_data(int degree) const {
  switch (degree) {
    case 0: return thrust::raw_pointer_cast(m_sh0.data());
    case 1: return thrust::raw_pointer_cast(m_sh1.data());
    case 2: return thrust::raw_pointer_cast(m_sh2.data());
    case 3: return thrust::raw_pointer_cast(m_sh3.data());
    default: return nullptr;
  }
}

// ============================================================================
// copy_from_host / copy_to_host: AoS <-> SoA conversion
// ============================================================================

void GPUGaussian3d::copy_from_host(const Gaussian3d& gaussians) {
  NVTX3_FUNC_RANGE();
  const int N = static_cast<int>(gaussians.means.size());

  // Resize and copy non-SH fields (direct copy, same layout)
  m_means.resize(N);
  m_opacities.resize(N);
  m_rotations.resize(N);
  m_scales.resize(N);

  thrust::copy(gaussians.means.begin(), gaussians.means.end(), m_means.begin());
  thrust::copy(gaussians.opacities.begin(), gaussians.opacities.end(), m_opacities.begin());
  thrust::copy(gaussians.rotations.begin(), gaussians.rotations.end(), m_rotations.begin());
  thrust::copy(gaussians.scales.begin(), gaussians.scales.end(), m_scales.begin());

  // SH: AoS (CPU) -> SoA (GPU) conversion per degree
  upload_sh_aos_to_soa(gaussians.sh0, m_sh0, N, 1);
  upload_sh_aos_to_soa(gaussians.sh1, m_sh1, N, 3);
  upload_sh_aos_to_soa(gaussians.sh2, m_sh2, N, 5);
  upload_sh_aos_to_soa(gaussians.sh3, m_sh3, N, 7);
}

void GPUGaussian3d::copy_to_host(Gaussian3d& gaussians) {
  NVTX3_FUNC_RANGE();
  const int N = static_cast<int>(m_means.size());

  // Resize and copy non-SH fields
  gaussians.means.resize(N);
  gaussians.opacities.resize(N);
  gaussians.rotations.resize(N);
  gaussians.scales.resize(N);

  thrust::copy(m_means.begin(), m_means.end(), gaussians.means.begin());
  thrust::copy(m_opacities.begin(), m_opacities.end(), gaussians.opacities.begin());
  thrust::copy(m_rotations.begin(), m_rotations.end(), gaussians.rotations.begin());
  thrust::copy(m_scales.begin(), m_scales.end(), gaussians.scales.begin());

  // SH: SoA (GPU) -> AoS (CPU) conversion per degree
  download_sh_soa_to_aos(m_sh0, gaussians.sh0, N, 1);
  download_sh_soa_to_aos(m_sh1, gaussians.sh1, N, 3);
  download_sh_soa_to_aos(m_sh2, gaussians.sh2, N, 5);
  download_sh_soa_to_aos(m_sh3, gaussians.sh3, N, 7);
}

// ============================================================================
// memset
// ============================================================================

void GPUGaussian3d::memset_async(char value, BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_means.data()), value, sizeof(float3) * m_means.size(), cuda_stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_opacities.data()), value, sizeof(float) * m_opacities.size(), cuda_stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_rotations.data()), value, sizeof(float4) * m_rotations.size(), cuda_stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_scales.data()), value, sizeof(float3) * m_scales.size(), cuda_stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_sh0.data()), value, sizeof(float) * m_sh0.size(), cuda_stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_sh1.data()), value, sizeof(float) * m_sh1.size(), cuda_stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_sh2.data()), value, sizeof(float) * m_sh2.size(), cuda_stream));
  CUDA_CHECK_THROW(cudaMemsetAsync(thrust::raw_pointer_cast(m_sh3.data()), value, sizeof(float) * m_sh3.size(), cuda_stream));
}

void GPUGaussian3d::memset(char value) {
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_means.data()), value, sizeof(float3) * m_means.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_opacities.data()), value, sizeof(float) * m_opacities.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_rotations.data()), value, sizeof(float4) * m_rotations.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_scales.data()), value, sizeof(float3) * m_scales.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_sh0.data()), value, sizeof(float) * m_sh0.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_sh1.data()), value, sizeof(float) * m_sh1.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_sh2.data()), value, sizeof(float) * m_sh2.size()));
  CUDA_CHECK_THROW(cudaMemset(thrust::raw_pointer_cast(m_sh3.data()), value, sizeof(float) * m_sh3.size()));
}

// ============================================================================
// Gather kernel for non-SH fields (base Gaussian data)
// ============================================================================

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

// ============================================================================
// Gather kernel for SoA SH buffers
//
// For a SH degree buffer with num_coeffs coefficients (SoA layout):
//   src element for Gaussian src_idx, coefficient k, channel c:
//     src[(k * 3 + c) * old_N + src_idx]
//   dst element for Gaussian dst_idx, coefficient k, channel c:
//     dst[(k * 3 + c) * new_N + dst_idx]
// ============================================================================

template<typename IndexType>
__global__ void gather_soa_sh_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const IndexType* __restrict__ mapping,
    int new_N,
    int old_N,
    int num_coeffs) {
  // One thread per (dst_gaussian, coefficient, channel)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = new_N * num_coeffs * 3;
  if (idx >= total) return;

  int i = idx % new_N;                     // Gaussian index in dst
  int kc = idx / new_N;                    // Combined (coeff * 3 + channel) index
  IndexType src_idx = mapping[i];

  dst[kc * new_N + i] = src[kc * old_N + src_idx];
}

// Re-layout helper for append: preserve first old_N items for each SoA channel
// while changing channel stride from old_N to new_N.
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

  int i = idx % old_N;     // Gaussian index in old layout
  int kc = idx / old_N;    // (coeff * 3 + channel)
  dst[kc * new_N + i] = src[kc * old_N + i];
}

/// @brief Helper to gather a single SoA SH buffer using a mapping.
template<typename IndexType>
static void gather_soa_sh(
    const thrust::device_vector<float>& src,
    thrust::device_vector<float>& dst,
    const IndexType* mapping,
    int new_N, int old_N, int num_coeffs,
    BackendStream stream = nullptr) {
  if (num_coeffs == 0 || new_N == 0) {
    dst.resize(num_coeffs * 3 * new_N);
    return;
  }
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  dst.resize(num_coeffs * 3 * new_N);
  int total = new_N * num_coeffs * 3;
  int blocks = (total + 255) / 256;
  gather_soa_sh_kernel<IndexType><<<blocks, 256, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(src.data()),
      thrust::raw_pointer_cast(dst.data()),
      mapping,
      new_N, old_N, num_coeffs);
}

// ============================================================================
// remove / append / reorder
// ============================================================================

void GPUGaussian3d::remove(char* kept_flag, int num_kept) {
  size_t original_size = size();
  thrust::device_vector<int> mapping(original_size);

  thrust::copy_if(
      thrust::device,
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(original_size),
      mapping.begin(),
      [kept_flag] __device__(int orig) { return static_cast<bool>(kept_flag[orig]); });

  // Create new vectors for base gaussian data
  thrust::device_vector<vec3> means(num_kept);
  thrust::device_vector<float> opacities(num_kept);
  thrust::device_vector<vec4> rotations(num_kept);
  thrust::device_vector<vec3> scales(num_kept);

  // Copy base items
  const int grid = (num_kept + 255) / 256;
  copy_gaussian_base_items<int><<<grid, 256>>>(
      thrust::raw_pointer_cast(m_means.data()),
      thrust::raw_pointer_cast(means.data()),
      thrust::raw_pointer_cast(m_opacities.data()),
      thrust::raw_pointer_cast(opacities.data()),
      thrust::raw_pointer_cast(m_rotations.data()),
      thrust::raw_pointer_cast(rotations.data()),
      thrust::raw_pointer_cast(m_scales.data()),
      thrust::raw_pointer_cast(scales.data()),
      thrust::raw_pointer_cast(mapping.data()),
      num_kept);

  // Gather SH SoA buffers
  thrust::device_vector<float> sh0_new, sh1_new, sh2_new, sh3_new;
  const int* map_ptr = thrust::raw_pointer_cast(mapping.data());
  gather_soa_sh(m_sh0, sh0_new, map_ptr, num_kept, (int)original_size, 1);
  gather_soa_sh(m_sh1, sh1_new, map_ptr, num_kept, (int)original_size, 3);
  gather_soa_sh(m_sh2, sh2_new, map_ptr, num_kept, (int)original_size, 5);
  gather_soa_sh(m_sh3, sh3_new, map_ptr, num_kept, (int)original_size, 7);

  // Move new vectors to replace old ones
  m_means = std::move(means);
  m_opacities = std::move(opacities);
  m_rotations = std::move(rotations);
  m_scales = std::move(scales);
  m_sh0 = std::move(sh0_new);
  m_sh1 = std::move(sh1_new);
  m_sh2 = std::move(sh2_new);
  m_sh3 = std::move(sh3_new);
}

void GPUGaussian3d::append(int num_dup) {
  assert(num_dup > 0);
  const size_t target_size = this->size() + static_cast<size_t>(num_dup);
  const size_t old_size = this->size();

  // Resize base fields
  m_means.resize(target_size, vec3(0.f));
  m_opacities.resize(target_size, 0.f);
  m_rotations.resize(target_size, vec4(0.f, 0.f, 0.f, 0.f));
  m_scales.resize(target_size, vec3(0.f, 0.f, 0.f));

  // For SoA SH buffers, we need to re-layout since N changed.
  // The old data at offsets (k*3+c)*old_N needs to move to (k*3+c)*new_N.
  // We use an identity mapping for the old gaussians and allocate zeros for the new.
  auto resize_soa_sh = [&](thrust::device_vector<float>& buf, int num_coeffs) {
    if (num_coeffs == 0) return;
    int new_total = num_coeffs * 3 * static_cast<int>(target_size);

    if (old_size == 0) {
      buf.resize(new_total, 0.f);
      return;
    }

    thrust::device_vector<float> new_buf(new_total, 0.f);
    int total_elems = static_cast<int>(old_size) * num_coeffs * 3;
    int blocks = (total_elems + 255) / 256;
    relayout_soa_sh_append_kernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(buf.data()),
        thrust::raw_pointer_cast(new_buf.data()),
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
}

// ============================================================================
// clone
// ============================================================================

/// @brief Async memcpy helper for a SoA SH buffer.
static void copy_sh_async(
    thrust::device_vector<float>& dst,
    const thrust::device_vector<float>& src,
    BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  dst.resize(src.size());
  if (src.empty()) return;
  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(dst.data()),
      thrust::raw_pointer_cast(src.data()),
      sizeof(float) * src.size(),
      cudaMemcpyDeviceToDevice,
      cuda_stream));
}

/// @brief Sync memcpy helper for a SoA SH buffer.
static void copy_sh_sync(
    thrust::device_vector<float>& dst,
    const thrust::device_vector<float>& src) {
  dst.resize(src.size());
  if (src.empty()) return;
  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(dst.data()),
      thrust::raw_pointer_cast(src.data()),
      sizeof(float) * src.size(),
      cudaMemcpyDeviceToDevice));
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone_async(BackendStream stream) {
  const cudaStream_t cuda_stream = to_cuda_stream(stream);
  auto gaussians = std::make_unique<GPUGaussian3d>();
  gaussians->m_current_sh_degree = m_current_sh_degree;
  gaussians->m_scene_scale = m_scene_scale;

  // Base fields
  gaussians->m_means.resize(m_means.size());
  gaussians->m_opacities.resize(m_opacities.size());
  gaussians->m_rotations.resize(m_rotations.size());
  gaussians->m_scales.resize(m_scales.size());

  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_means.data()),
      thrust::raw_pointer_cast(m_means.data()),
      sizeof(float3) * m_means.size(), cudaMemcpyDeviceToDevice, cuda_stream));
  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_opacities.data()),
      thrust::raw_pointer_cast(m_opacities.data()),
      sizeof(float) * m_opacities.size(), cudaMemcpyDeviceToDevice, cuda_stream));
  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_rotations.data()),
      thrust::raw_pointer_cast(m_rotations.data()),
      sizeof(float4) * m_rotations.size(), cudaMemcpyDeviceToDevice, cuda_stream));
  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(gaussians->m_scales.data()),
      thrust::raw_pointer_cast(m_scales.data()),
      sizeof(float3) * m_scales.size(), cudaMemcpyDeviceToDevice, cuda_stream));

  // SH buffers
  copy_sh_async(gaussians->m_sh0, m_sh0, stream);
  copy_sh_async(gaussians->m_sh1, m_sh1, stream);
  copy_sh_async(gaussians->m_sh2, m_sh2, stream);
  copy_sh_async(gaussians->m_sh3, m_sh3, stream);

  return gaussians;
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone() {
  auto gaussians = std::make_unique<GPUGaussian3d>();
  gaussians->m_current_sh_degree = m_current_sh_degree;
  gaussians->m_scene_scale = m_scene_scale;

  // Base fields
  gaussians->m_means.resize(m_means.size());
  gaussians->m_opacities.resize(m_opacities.size());
  gaussians->m_rotations.resize(m_rotations.size());
  gaussians->m_scales.resize(m_scales.size());

  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_means.data()),
      thrust::raw_pointer_cast(m_means.data()),
      sizeof(float3) * m_means.size(), cudaMemcpyDeviceToDevice));
  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_opacities.data()),
      thrust::raw_pointer_cast(m_opacities.data()),
      sizeof(float) * m_opacities.size(), cudaMemcpyDeviceToDevice));
  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_rotations.data()),
      thrust::raw_pointer_cast(m_rotations.data()),
      sizeof(float4) * m_rotations.size(), cudaMemcpyDeviceToDevice));
  CUDA_CHECK_THROW(cudaMemcpy(
      thrust::raw_pointer_cast(gaussians->m_scales.data()),
      thrust::raw_pointer_cast(m_scales.data()),
      sizeof(float3) * m_scales.size(), cudaMemcpyDeviceToDevice));

  // SH buffers
  copy_sh_sync(gaussians->m_sh0, m_sh0);
  copy_sh_sync(gaussians->m_sh1, m_sh1);
  copy_sh_sync(gaussians->m_sh2, m_sh2);
  copy_sh_sync(gaussians->m_sh3, m_sh3);

  return gaussians;
}

// ============================================================================
// reorder
// ============================================================================

void GPUGaussian3d::reorder(uint* indices, BackendStream stream) {
  NVTX3_FUNC_RANGE();
  const int N = static_cast<int>(size());
  const cudaStream_t cuda_stream = to_cuda_stream(stream);

  // Create temporary vectors for base fields
  thrust::device_vector<vec3> means(N);
  thrust::device_vector<float> opacities(N);
  thrust::device_vector<vec4> rotations(N);
  thrust::device_vector<vec3> scales(N);

  const int grid = (N + 255) / 256;
  copy_gaussian_base_items<uint><<<grid, 256, 0, cuda_stream>>>(
      thrust::raw_pointer_cast(m_means.data()),
      thrust::raw_pointer_cast(means.data()),
      thrust::raw_pointer_cast(m_opacities.data()),
      thrust::raw_pointer_cast(opacities.data()),
      thrust::raw_pointer_cast(m_rotations.data()),
      thrust::raw_pointer_cast(rotations.data()),
      thrust::raw_pointer_cast(m_scales.data()),
      thrust::raw_pointer_cast(scales.data()),
      indices, N);

  // Reorder SH SoA buffers
  thrust::device_vector<float> sh0_new, sh1_new, sh2_new, sh3_new;
  gather_soa_sh(m_sh0, sh0_new, indices, N, N, 1, stream);
  gather_soa_sh(m_sh1, sh1_new, indices, N, N, 3, stream);
  gather_soa_sh(m_sh2, sh2_new, indices, N, N, 5, stream);
  gather_soa_sh(m_sh3, sh3_new, indices, N, N, 7, stream);

  // Move
  m_means = std::move(means);
  m_opacities = std::move(opacities);
  m_rotations = std::move(rotations);
  m_scales = std::move(scales);
  m_sh0 = std::move(sh0_new);
  m_sh1 = std::move(sh1_new);
  m_sh2 = std::move(sh2_new);
  m_sh3 = std::move(sh3_new);
}

#undef m_means
#undef m_opacities
#undef m_rotations
#undef m_scales
#undef m_sh0
#undef m_sh1
#undef m_sh2
#undef m_sh3

}  // namespace tinygs
