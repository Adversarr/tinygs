#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/platform/buffer_utils.hpp"
#include "tinygs/platform/runtime.hpp"

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

namespace {

std::shared_ptr<BackendQueue> create_internal_queue(const std::shared_ptr<BackendRuntime>& runtime) {
  QueueDesc queue_desc;
  queue_desc.non_blocking = true;
  auto queue_result = runtime->create_queue(queue_desc);
  detail::throw_if_status_error(queue_result.error(), "create_queue");
  return queue_result.value();
}

std::shared_ptr<BackendQueue> make_effective_queue(
    const std::shared_ptr<BackendRuntime>& runtime,
    const BackendQueue* queue) {
  if (queue != nullptr) {
    return std::shared_ptr<BackendQueue>(const_cast<BackendQueue*>(queue), [](BackendQueue*) {});
  }
  return create_internal_queue(runtime);
}

std::shared_ptr<BackendBuffer> clone_buffer_direct(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendBuffer>& src,
    const char* debug_name) {
  if (!src || src->size_bytes() == 0) {
    return nullptr;
  }
  auto dst = create_device_buffer(*runtime, src->size_bytes(), debug_name);
  std::memcpy(buffer_data<void>(dst), buffer_data_const<void>(src), src->size_bytes());
  return dst;
}

template <typename T>
std::shared_ptr<BackendBuffer> create_buffer_or_reset(
    const std::shared_ptr<BackendRuntime>& runtime,
    size_t count,
    const char* debug_name) {
  if (count == 0) {
    return nullptr;
  }
  return create_device_buffer_for<T>(*runtime, count, debug_name);
}

template <typename T>
void copy_buffer_range(
    const std::shared_ptr<BackendBuffer>& src,
    const std::shared_ptr<BackendBuffer>& dst,
    size_t count,
    const std::vector<size_t>& mapping) {
  const T* src_ptr = buffer_data_const<T>(src);
  T* dst_ptr = buffer_data<T>(dst);
  for (size_t i = 0; i < count; ++i) {
    dst_ptr[i] = src_ptr[mapping[i]];
  }
}

void gather_sh_soa(
    const std::shared_ptr<BackendBuffer>& src,
    std::shared_ptr<BackendBuffer>& dst,
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::vector<size_t>& mapping,
    int new_n,
    int old_n,
    int num_coeffs) {
  const int total = new_n * num_coeffs * 3;
  dst = create_buffer_or_reset<float>(runtime, static_cast<size_t>(total), "sh_gather");
  if (total == 0) {
    return;
  }
  const float* src_ptr = buffer_data_const<float>(src);
  float* dst_ptr = buffer_data<float>(dst);
  for (int kc = 0; kc < num_coeffs * 3; ++kc) {
    const size_t src_base = static_cast<size_t>(kc) * static_cast<size_t>(old_n);
    const size_t dst_base = static_cast<size_t>(kc) * static_cast<size_t>(new_n);
    for (int i = 0; i < new_n; ++i) {
      dst_ptr[dst_base + static_cast<size_t>(i)] = src_ptr[src_base + mapping[static_cast<size_t>(i)]];
    }
  }
}

void upload_sh_aos_to_soa(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::vector<vec3>& host_aos,
    std::shared_ptr<BackendBuffer>& gpu_soa,
    int n,
    int num_coeffs) {
  if (n == 0 || num_coeffs == 0) {
    gpu_soa.reset();
    return;
  }
  CHECK_THROW(queue != nullptr);
  CHECK_THROW(static_cast<int>(host_aos.size()) == n * num_coeffs);
  std::vector<float> host_soa(static_cast<size_t>(n) * static_cast<size_t>(num_coeffs) * 3ull);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < num_coeffs; ++k) {
      const vec3 val = host_aos[static_cast<size_t>(i) * static_cast<size_t>(num_coeffs) + static_cast<size_t>(k)];
      host_soa[(static_cast<size_t>(k) * 3ull + 0ull) * static_cast<size_t>(n) + static_cast<size_t>(i)] = val.x;
      host_soa[(static_cast<size_t>(k) * 3ull + 1ull) * static_cast<size_t>(n) + static_cast<size_t>(i)] = val.y;
      host_soa[(static_cast<size_t>(k) * 3ull + 2ull) * static_cast<size_t>(n) + static_cast<size_t>(i)] = val.z;
    }
  }
  gpu_soa = create_device_buffer_for<float>(*runtime, host_soa.size(), "sh_soa");
  tinygs::copy_from_host_async(*runtime, *queue, gpu_soa, host_soa.data(), host_soa.size());
}

void download_sh_soa_to_aos(
    const std::shared_ptr<BackendRuntime>& runtime,
    const std::shared_ptr<BackendQueue>& queue,
    const std::shared_ptr<BackendBuffer>& gpu_soa,
    std::vector<vec3>& host_aos,
    int n,
    int num_coeffs) {
  if (n == 0 || num_coeffs == 0) {
    host_aos.clear();
    return;
  }
  CHECK_THROW(queue != nullptr);
  std::vector<float> host_soa(static_cast<size_t>(n) * static_cast<size_t>(num_coeffs) * 3ull);
  tinygs::copy_to_host_async(*runtime, *queue, gpu_soa, host_soa.data(), host_soa.size());
  host_aos.resize(static_cast<size_t>(n) * static_cast<size_t>(num_coeffs));
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < num_coeffs; ++k) {
      vec3 val;
      val.x = host_soa[(static_cast<size_t>(k) * 3ull + 0ull) * static_cast<size_t>(n) + static_cast<size_t>(i)];
      val.y = host_soa[(static_cast<size_t>(k) * 3ull + 1ull) * static_cast<size_t>(n) + static_cast<size_t>(i)];
      val.z = host_soa[(static_cast<size_t>(k) * 3ull + 2ull) * static_cast<size_t>(n) + static_cast<size_t>(i)];
      host_aos[static_cast<size_t>(i) * static_cast<size_t>(num_coeffs) + static_cast<size_t>(k)] = val;
    }
  }
}

uint32_t expand_bits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

uint32_t morton3d(uint32_t x, uint32_t y, uint32_t z) {
  const uint32_t xx = expand_bits(x);
  const uint32_t yy = expand_bits(y);
  const uint32_t zz = expand_bits(z);
  return xx | (yy << 1) | (zz << 2);
}

}  // namespace

GPUGaussian3d::GPUGaussian3d(std::shared_ptr<BackendRuntime> runtime)
    : m_impl(std::make_unique<Impl>()) {
  m_impl->m_runtime = std::move(runtime);
}

GPUGaussian3d::~GPUGaussian3d() = default;

std::shared_ptr<BackendRuntime> GPUGaussian3d::runtime() const {
  return m_impl->m_runtime;
}

size_t GPUGaussian3d::size() const {
  return m_impl->m_size;
}

DeviceSpan<const vec3> GPUGaussian3d::means() const {
  return DeviceSpan<const vec3>{m_impl->m_means.get(), 0, m_impl->m_size};
}

DeviceSpan<const float> GPUGaussian3d::opacities() const {
  return DeviceSpan<const float>{m_impl->m_opacities.get(), 0, m_impl->m_size};
}

DeviceSpan<const vec4> GPUGaussian3d::rotations() const {
  return DeviceSpan<const vec4>{m_impl->m_rotations.get(), 0, m_impl->m_size};
}

DeviceSpan<const vec3> GPUGaussian3d::scales() const {
  return DeviceSpan<const vec3>{m_impl->m_scales.get(), 0, m_impl->m_size};
}

DeviceSpan<vec3> GPUGaussian3d::means() {
  return DeviceSpan<vec3>{m_impl->m_means.get(), 0, m_impl->m_size};
}

DeviceSpan<float> GPUGaussian3d::opacities() {
  return DeviceSpan<float>{m_impl->m_opacities.get(), 0, m_impl->m_size};
}

DeviceSpan<vec4> GPUGaussian3d::rotations() {
  return DeviceSpan<vec4>{m_impl->m_rotations.get(), 0, m_impl->m_size};
}

DeviceSpan<vec3> GPUGaussian3d::scales() {
  return DeviceSpan<vec3>{m_impl->m_scales.get(), 0, m_impl->m_size};
}

DeviceSpan<const float> GPUGaussian3d::sh0() const {
  return DeviceSpan<const float>{m_impl->m_sh0.get()};
}

DeviceSpan<const float> GPUGaussian3d::sh1() const {
  return DeviceSpan<const float>{m_impl->m_sh1.get()};
}

DeviceSpan<const float> GPUGaussian3d::sh2() const {
  return DeviceSpan<const float>{m_impl->m_sh2.get()};
}

DeviceSpan<const float> GPUGaussian3d::sh3() const {
  return DeviceSpan<const float>{m_impl->m_sh3.get()};
}

DeviceSpan<float> GPUGaussian3d::sh0() {
  return DeviceSpan<float>{m_impl->m_sh0.get()};
}

DeviceSpan<float> GPUGaussian3d::sh1() {
  return DeviceSpan<float>{m_impl->m_sh1.get()};
}

DeviceSpan<float> GPUGaussian3d::sh2() {
  return DeviceSpan<float>{m_impl->m_sh2.get()};
}

DeviceSpan<float> GPUGaussian3d::sh3() {
  return DeviceSpan<float>{m_impl->m_sh3.get()};
}

float* GPUGaussian3d::sh_degree_data(int degree) {
  switch (degree) {
    case 0: return buffer_data<float>(m_impl->m_sh0);
    case 1: return buffer_data<float>(m_impl->m_sh1);
    case 2: return buffer_data<float>(m_impl->m_sh2);
    case 3: return buffer_data<float>(m_impl->m_sh3);
    default: return nullptr;
  }
}

const float* GPUGaussian3d::sh_degree_data(int degree) const {
  switch (degree) {
    case 0: return buffer_data_const<float>(m_impl->m_sh0);
    case 1: return buffer_data_const<float>(m_impl->m_sh1);
    case 2: return buffer_data_const<float>(m_impl->m_sh2);
    case 3: return buffer_data_const<float>(m_impl->m_sh3);
    default: return nullptr;
  }
}

void GPUGaussian3d::copy_from_host_async(
    const Gaussian3d& gaussians,
    const std::shared_ptr<BackendQueue>& queue) {
  CHECK_THROW(queue != nullptr);
  const int n = static_cast<int>(gaussians.means.size());
  CHECK_THROW(static_cast<int>(gaussians.opacities.size()) == n);
  CHECK_THROW(static_cast<int>(gaussians.rotations.size()) == n);
  CHECK_THROW(static_cast<int>(gaussians.scales.size()) == n);
  CHECK_THROW(static_cast<int>(gaussians.sh0.size()) == n * 1);
  CHECK_THROW(static_cast<int>(gaussians.sh1.size()) == n * 3);
  CHECK_THROW(static_cast<int>(gaussians.sh2.size()) == n * 5);
  CHECK_THROW(static_cast<int>(gaussians.sh3.size()) == n * 7);

  m_impl->m_size = static_cast<size_t>(n);
  m_impl->m_means = create_buffer_or_reset<vec3>(m_impl->m_runtime, m_impl->m_size, "means");
  m_impl->m_opacities = create_buffer_or_reset<float>(m_impl->m_runtime, m_impl->m_size, "opacities");
  m_impl->m_rotations = create_buffer_or_reset<vec4>(m_impl->m_runtime, m_impl->m_size, "rotations");
  m_impl->m_scales = create_buffer_or_reset<vec3>(m_impl->m_runtime, m_impl->m_size, "scales");

  if (m_impl->m_size > 0) {
    tinygs::copy_from_host_async(*m_impl->m_runtime, *queue, m_impl->m_means, gaussians.means.data(), m_impl->m_size);
    tinygs::copy_from_host_async(
        m_impl->m_runtime, queue, m_impl->m_opacities, gaussians.opacities.data(), m_impl->m_size);
    tinygs::copy_from_host_async(
        m_impl->m_runtime, queue, m_impl->m_rotations, gaussians.rotations.data(), m_impl->m_size);
    tinygs::copy_from_host_async(*m_impl->m_runtime, *queue, m_impl->m_scales, gaussians.scales.data(), m_impl->m_size);
  }

  upload_sh_aos_to_soa(m_impl->m_runtime, queue, gaussians.sh0, m_impl->m_sh0, n, 1);
  upload_sh_aos_to_soa(m_impl->m_runtime, queue, gaussians.sh1, m_impl->m_sh1, n, 3);
  upload_sh_aos_to_soa(m_impl->m_runtime, queue, gaussians.sh2, m_impl->m_sh2, n, 5);
  upload_sh_aos_to_soa(m_impl->m_runtime, queue, gaussians.sh3, m_impl->m_sh3, n, 7);
}

void GPUGaussian3d::copy_from_host(
    const Gaussian3d& gaussians,
    const std::shared_ptr<BackendQueue>& queue) {
  copy_from_host_async(gaussians, queue);
  detail::throw_if_status_error(m_impl->m_runtime->synchronize_queue(*queue), "GPUGaussian3d::copy_from_host sync");
}

void GPUGaussian3d::copy_to_host_async(
    Gaussian3d& gaussians,
    const std::shared_ptr<BackendQueue>& queue) {
  CHECK_THROW(queue != nullptr);
  const int n = static_cast<int>(m_impl->m_size);
  gaussians.means.resize(m_impl->m_size);
  gaussians.opacities.resize(m_impl->m_size);
  gaussians.rotations.resize(m_impl->m_size);
  gaussians.scales.resize(m_impl->m_size);
  if (m_impl->m_size > 0) {
    tinygs::copy_to_host_async(*m_impl->m_runtime, *queue, m_impl->m_means, gaussians.means.data(), m_impl->m_size);
    tinygs::copy_to_host_async(
        m_impl->m_runtime, queue, m_impl->m_opacities, gaussians.opacities.data(), m_impl->m_size);
    tinygs::copy_to_host_async(
        m_impl->m_runtime, queue, m_impl->m_rotations, gaussians.rotations.data(), m_impl->m_size);
    tinygs::copy_to_host_async(*m_impl->m_runtime, *queue, m_impl->m_scales, gaussians.scales.data(), m_impl->m_size);
  }
  download_sh_soa_to_aos(m_impl->m_runtime, queue, m_impl->m_sh0, gaussians.sh0, n, 1);
  download_sh_soa_to_aos(m_impl->m_runtime, queue, m_impl->m_sh1, gaussians.sh1, n, 3);
  download_sh_soa_to_aos(m_impl->m_runtime, queue, m_impl->m_sh2, gaussians.sh2, n, 5);
  download_sh_soa_to_aos(m_impl->m_runtime, queue, m_impl->m_sh3, gaussians.sh3, n, 7);
}

void GPUGaussian3d::copy_to_host(
    Gaussian3d& gaussians,
    const std::shared_ptr<BackendQueue>& queue) {
  copy_to_host_async(gaussians, queue);
  detail::throw_if_status_error(m_impl->m_runtime->synchronize_queue(*queue), "GPUGaussian3d::copy_to_host sync");
}

void GPUGaussian3d::memset_async(char value, const BackendQueue* queue) {
  const auto effective_queue = make_effective_queue(m_impl->m_runtime, queue);
  const uint8_t byte_value = static_cast<uint8_t>(value);
  if (m_impl->m_means && m_impl->m_means->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_means, byte_value);
  if (m_impl->m_opacities && m_impl->m_opacities->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_opacities, byte_value);
  if (m_impl->m_rotations && m_impl->m_rotations->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_rotations, byte_value);
  if (m_impl->m_scales && m_impl->m_scales->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_scales, byte_value);
  if (m_impl->m_sh0 && m_impl->m_sh0->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_sh0, byte_value);
  if (m_impl->m_sh1 && m_impl->m_sh1->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_sh1, byte_value);
  if (m_impl->m_sh2 && m_impl->m_sh2->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_sh2, byte_value);
  if (m_impl->m_sh3 && m_impl->m_sh3->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, effective_queue, m_impl->m_sh3, byte_value);
}

void GPUGaussian3d::memset(char value) {
  const auto queue = create_internal_queue(m_impl->m_runtime);
  const uint8_t byte_value = static_cast<uint8_t>(value);
  if (m_impl->m_means && m_impl->m_means->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_means, byte_value);
  if (m_impl->m_opacities && m_impl->m_opacities->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_opacities, byte_value);
  if (m_impl->m_rotations && m_impl->m_rotations->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_rotations, byte_value);
  if (m_impl->m_scales && m_impl->m_scales->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_scales, byte_value);
  if (m_impl->m_sh0 && m_impl->m_sh0->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_sh0, byte_value);
  if (m_impl->m_sh1 && m_impl->m_sh1->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_sh1, byte_value);
  if (m_impl->m_sh2 && m_impl->m_sh2->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_sh2, byte_value);
  if (m_impl->m_sh3 && m_impl->m_sh3->size_bytes() > 0) fill_buffer_async(m_impl->m_runtime, queue, m_impl->m_sh3, byte_value);
  detail::throw_if_status_error(m_impl->m_runtime->synchronize_queue(*queue), "GPUGaussian3d::memset sync");
}

void GPUGaussian3d::remove(char* kept_flag, int num_kept, const BackendQueue* queue) {
  CHECK_THROW(num_kept >= 0);
  const auto effective_queue = make_effective_queue(m_impl->m_runtime, queue);
  detail::throw_if_status_error(
      m_impl->m_runtime->synchronize_queue(*effective_queue),
      "GPUGaussian3d::remove sync");
  const size_t old_size = m_impl->m_size;
  if (old_size == 0) {
    CHECK_THROW(num_kept == 0);
    m_impl->m_size = 0;
    m_impl->m_means.reset();
    m_impl->m_opacities.reset();
    m_impl->m_rotations.reset();
    m_impl->m_scales.reset();
    m_impl->m_sh0.reset();
    m_impl->m_sh1.reset();
    m_impl->m_sh2.reset();
    m_impl->m_sh3.reset();
    return;
  }
  CHECK_THROW(kept_flag != nullptr);
  std::vector<size_t> mapping;
  mapping.reserve(static_cast<size_t>(num_kept));
  for (size_t i = 0; i < old_size; ++i) {
    if (kept_flag[i]) {
      mapping.push_back(i);
    }
  }
  CHECK_THROW(mapping.size() == static_cast<size_t>(num_kept));

  auto new_means = create_buffer_or_reset<vec3>(m_impl->m_runtime, mapping.size(), "means");
  auto new_opacities = create_buffer_or_reset<float>(m_impl->m_runtime, mapping.size(), "opacities");
  auto new_rotations = create_buffer_or_reset<vec4>(m_impl->m_runtime, mapping.size(), "rotations");
  auto new_scales = create_buffer_or_reset<vec3>(m_impl->m_runtime, mapping.size(), "scales");
  if (!mapping.empty()) {
    copy_buffer_range<vec3>(m_impl->m_means, new_means, mapping.size(), mapping);
    copy_buffer_range<float>(m_impl->m_opacities, new_opacities, mapping.size(), mapping);
    copy_buffer_range<vec4>(m_impl->m_rotations, new_rotations, mapping.size(), mapping);
    copy_buffer_range<vec3>(m_impl->m_scales, new_scales, mapping.size(), mapping);
  }

  std::shared_ptr<BackendBuffer> sh0_new;
  std::shared_ptr<BackendBuffer> sh1_new;
  std::shared_ptr<BackendBuffer> sh2_new;
  std::shared_ptr<BackendBuffer> sh3_new;
  gather_sh_soa(m_impl->m_sh0, sh0_new, m_impl->m_runtime, mapping, num_kept, static_cast<int>(old_size), 1);
  gather_sh_soa(m_impl->m_sh1, sh1_new, m_impl->m_runtime, mapping, num_kept, static_cast<int>(old_size), 3);
  gather_sh_soa(m_impl->m_sh2, sh2_new, m_impl->m_runtime, mapping, num_kept, static_cast<int>(old_size), 5);
  gather_sh_soa(m_impl->m_sh3, sh3_new, m_impl->m_runtime, mapping, num_kept, static_cast<int>(old_size), 7);

  m_impl->m_means = std::move(new_means);
  m_impl->m_opacities = std::move(new_opacities);
  m_impl->m_rotations = std::move(new_rotations);
  m_impl->m_scales = std::move(new_scales);
  m_impl->m_sh0 = std::move(sh0_new);
  m_impl->m_sh1 = std::move(sh1_new);
  m_impl->m_sh2 = std::move(sh2_new);
  m_impl->m_sh3 = std::move(sh3_new);
  m_impl->m_size = static_cast<size_t>(num_kept);
}

void GPUGaussian3d::append(int num_dup, const std::shared_ptr<BackendQueue>& queue) {
  CHECK_THROW(queue != nullptr);
  CHECK_THROW(num_dup > 0);
  const size_t old_size = m_impl->m_size;
  const size_t target_size = old_size + static_cast<size_t>(num_dup);

  auto new_means = create_buffer_or_reset<vec3>(m_impl->m_runtime, target_size, "means");
  auto new_opacities = create_buffer_or_reset<float>(m_impl->m_runtime, target_size, "opacities");
  auto new_rotations = create_buffer_or_reset<vec4>(m_impl->m_runtime, target_size, "rotations");
  auto new_scales = create_buffer_or_reset<vec3>(m_impl->m_runtime, target_size, "scales");
  if (target_size > 0) {
    fill_buffer_zero_async(*m_impl->m_runtime, *queue, new_means);
    fill_buffer_zero_async(*m_impl->m_runtime, *queue, new_opacities);
    fill_buffer_zero_async(*m_impl->m_runtime, *queue, new_rotations);
    fill_buffer_zero_async(*m_impl->m_runtime, *queue, new_scales);
  }
  if (old_size > 0) {
    copy_buffer_async(*m_impl->m_runtime, *queue, new_means, m_impl->m_means, old_size * sizeof(vec3));
    copy_buffer_async(*m_impl->m_runtime, *queue, new_opacities, m_impl->m_opacities, old_size * sizeof(float));
    copy_buffer_async(*m_impl->m_runtime, *queue, new_rotations, m_impl->m_rotations, old_size * sizeof(vec4));
    copy_buffer_async(*m_impl->m_runtime, *queue, new_scales, m_impl->m_scales, old_size * sizeof(vec3));
  }

  auto resize_soa_sh = [&](std::shared_ptr<BackendBuffer>& buf, int num_coeffs) {
    const int new_total = num_coeffs * 3 * static_cast<int>(target_size);
    auto new_buf = create_buffer_or_reset<float>(m_impl->m_runtime, static_cast<size_t>(new_total), "sh_resize");
    if (new_total > 0) {
      fill_buffer_zero_async(*m_impl->m_runtime, *queue, new_buf);
    }
    if (old_size > 0 && buf && buf->size_bytes() > 0) {
      const float* src_ptr = buffer_data_const<float>(buf);
      float* dst_ptr = buffer_data<float>(new_buf);
      const int channels = num_coeffs * 3;
      for (int kc = 0; kc < channels; ++kc) {
        const size_t src_base = static_cast<size_t>(kc) * old_size;
        const size_t dst_base = static_cast<size_t>(kc) * target_size;
        std::memcpy(dst_ptr + dst_base, src_ptr + src_base, old_size * sizeof(float));
      }
    }
    buf = std::move(new_buf);
  };

  resize_soa_sh(m_impl->m_sh0, 1);
  resize_soa_sh(m_impl->m_sh1, 3);
  resize_soa_sh(m_impl->m_sh2, 5);
  resize_soa_sh(m_impl->m_sh3, 7);

  m_impl->m_means = std::move(new_means);
  m_impl->m_opacities = std::move(new_opacities);
  m_impl->m_rotations = std::move(new_rotations);
  m_impl->m_scales = std::move(new_scales);
  m_impl->m_size = target_size;
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone_async(const BackendQueue* queue) {
  const auto effective_queue = make_effective_queue(m_impl->m_runtime, queue);
  detail::throw_if_status_error(
      m_impl->m_runtime->synchronize_queue(*effective_queue),
      "GPUGaussian3d::clone_async sync");
  auto gaussians = std::make_unique<GPUGaussian3d>(m_impl->m_runtime);
  gaussians->m_current_sh_degree = m_current_sh_degree;
  gaussians->m_scene_scale = m_scene_scale;
  gaussians->m_impl->m_size = m_impl->m_size;
  gaussians->m_impl->m_means = clone_buffer_direct(m_impl->m_runtime, m_impl->m_means, "means");
  gaussians->m_impl->m_opacities = clone_buffer_direct(m_impl->m_runtime, m_impl->m_opacities, "opacities");
  gaussians->m_impl->m_rotations = clone_buffer_direct(m_impl->m_runtime, m_impl->m_rotations, "rotations");
  gaussians->m_impl->m_scales = clone_buffer_direct(m_impl->m_runtime, m_impl->m_scales, "scales");
  gaussians->m_impl->m_sh0 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh0, "sh0");
  gaussians->m_impl->m_sh1 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh1, "sh1");
  gaussians->m_impl->m_sh2 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh2, "sh2");
  gaussians->m_impl->m_sh3 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh3, "sh3");
  return gaussians;
}

std::unique_ptr<GPUGaussian3d> GPUGaussian3d::clone() {
  auto gaussians = std::make_unique<GPUGaussian3d>(m_impl->m_runtime);
  gaussians->m_current_sh_degree = m_current_sh_degree;
  gaussians->m_scene_scale = m_scene_scale;
  gaussians->m_impl->m_size = m_impl->m_size;
  gaussians->m_impl->m_means = clone_buffer_direct(m_impl->m_runtime, m_impl->m_means, "means");
  gaussians->m_impl->m_opacities = clone_buffer_direct(m_impl->m_runtime, m_impl->m_opacities, "opacities");
  gaussians->m_impl->m_rotations = clone_buffer_direct(m_impl->m_runtime, m_impl->m_rotations, "rotations");
  gaussians->m_impl->m_scales = clone_buffer_direct(m_impl->m_runtime, m_impl->m_scales, "scales");
  gaussians->m_impl->m_sh0 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh0, "sh0");
  gaussians->m_impl->m_sh1 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh1, "sh1");
  gaussians->m_impl->m_sh2 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh2, "sh2");
  gaussians->m_impl->m_sh3 = clone_buffer_direct(m_impl->m_runtime, m_impl->m_sh3, "sh3");
  return gaussians;
}

std::shared_ptr<BackendBuffer> GPUGaussian3d::compute_morton_order_indices(const BackendQueue* queue) {
  const auto effective_queue = make_effective_queue(m_impl->m_runtime, queue);
  detail::throw_if_status_error(
      m_impl->m_runtime->synchronize_queue(*effective_queue),
      "GPUGaussian3d::compute_morton_order_indices sync");
  const uint32_t n = static_cast<uint32_t>(m_impl->m_size);
  if (n == 0) {
    return nullptr;
  }
  const vec3* positions = buffer_data_const<vec3>(m_impl->m_means);
  CHECK_THROW(positions != nullptr);

  vec3 min_pos(FLT_MAX, FLT_MAX, FLT_MAX);
  vec3 max_pos(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  for (uint32_t i = 0; i < n; ++i) {
    const vec3 p = positions[i];
    min_pos.x = std::min(min_pos.x, p.x);
    min_pos.y = std::min(min_pos.y, p.y);
    min_pos.z = std::min(min_pos.z, p.z);
    max_pos.x = std::max(max_pos.x, p.x);
    max_pos.y = std::max(max_pos.y, p.y);
    max_pos.z = std::max(max_pos.z, p.z);
  }

  const float inv_dx = 1.0f / std::max(max_pos.x - min_pos.x, 1e-8f);
  const float inv_dy = 1.0f / std::max(max_pos.y - min_pos.y, 1e-8f);
  const float inv_dz = 1.0f / std::max(max_pos.z - min_pos.z, 1e-8f);

  std::vector<uint32_t> indices(static_cast<size_t>(n));
  std::iota(indices.begin(), indices.end(), 0u);
  std::vector<uint32_t> morton_codes(static_cast<size_t>(n));
  for (uint32_t i = 0; i < n; ++i) {
    const vec3 p = positions[i];
    const float nx = std::clamp((p.x - min_pos.x) * inv_dx, 0.0f, 1.0f);
    const float ny = std::clamp((p.y - min_pos.y) * inv_dy, 0.0f, 1.0f);
    const float nz = std::clamp((p.z - min_pos.z) * inv_dz, 0.0f, 1.0f);
    const uint32_t xi = static_cast<uint32_t>(nx * 1023.0f);
    const uint32_t yi = static_cast<uint32_t>(ny * 1023.0f);
    const uint32_t zi = static_cast<uint32_t>(nz * 1023.0f);
    morton_codes[static_cast<size_t>(i)] = morton3d(xi, yi, zi);
  }

  std::stable_sort(indices.begin(), indices.end(), [&](uint32_t lhs, uint32_t rhs) {
    const uint32_t a = morton_codes[static_cast<size_t>(lhs)];
    const uint32_t b = morton_codes[static_cast<size_t>(rhs)];
    if (a != b) {
      return a < b;
    }
    return lhs < rhs;
  });

  auto idx_out = create_device_buffer_for<uint>(*m_impl->m_runtime, static_cast<size_t>(n), "morton_idx_out");
  tinygs::copy_from_host_async(m_impl->m_runtime, effective_queue, idx_out, indices.data(), indices.size());
  return idx_out;
}

void GPUGaussian3d::reorder(uint* indices, const BackendQueue* queue) {
  const auto effective_queue = make_effective_queue(m_impl->m_runtime, queue);
  detail::throw_if_status_error(
      m_impl->m_runtime->synchronize_queue(*effective_queue),
      "GPUGaussian3d::reorder sync");
  const int n = static_cast<int>(m_impl->m_size);
  if (n == 0) {
    return;
  }
  CHECK_THROW(indices != nullptr);

  std::vector<size_t> mapping(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    mapping[static_cast<size_t>(i)] = static_cast<size_t>(indices[i]);
    CHECK_THROW(mapping[static_cast<size_t>(i)] < static_cast<size_t>(n));
  }

  auto new_means = create_device_buffer_for<vec3>(*m_impl->m_runtime, static_cast<size_t>(n), "means");
  auto new_opacities = create_device_buffer_for<float>(*m_impl->m_runtime, static_cast<size_t>(n), "opacities");
  auto new_rotations = create_device_buffer_for<vec4>(*m_impl->m_runtime, static_cast<size_t>(n), "rotations");
  auto new_scales = create_device_buffer_for<vec3>(*m_impl->m_runtime, static_cast<size_t>(n), "scales");
  copy_buffer_range<vec3>(m_impl->m_means, new_means, static_cast<size_t>(n), mapping);
  copy_buffer_range<float>(m_impl->m_opacities, new_opacities, static_cast<size_t>(n), mapping);
  copy_buffer_range<vec4>(m_impl->m_rotations, new_rotations, static_cast<size_t>(n), mapping);
  copy_buffer_range<vec3>(m_impl->m_scales, new_scales, static_cast<size_t>(n), mapping);

  std::shared_ptr<BackendBuffer> sh0_new;
  std::shared_ptr<BackendBuffer> sh1_new;
  std::shared_ptr<BackendBuffer> sh2_new;
  std::shared_ptr<BackendBuffer> sh3_new;
  gather_sh_soa(m_impl->m_sh0, sh0_new, m_impl->m_runtime, mapping, n, n, 1);
  gather_sh_soa(m_impl->m_sh1, sh1_new, m_impl->m_runtime, mapping, n, n, 3);
  gather_sh_soa(m_impl->m_sh2, sh2_new, m_impl->m_runtime, mapping, n, n, 5);
  gather_sh_soa(m_impl->m_sh3, sh3_new, m_impl->m_runtime, mapping, n, n, 7);

  m_impl->m_means = std::move(new_means);
  m_impl->m_opacities = std::move(new_opacities);
  m_impl->m_rotations = std::move(new_rotations);
  m_impl->m_scales = std::move(new_scales);
  m_impl->m_sh0 = std::move(sh0_new);
  m_impl->m_sh1 = std::move(sh1_new);
  m_impl->m_sh2 = std::move(sh2_new);
  m_impl->m_sh3 = std::move(sh3_new);
}

std::shared_ptr<BackendBuffer> reorder_densification_info(
    const std::shared_ptr<BackendBuffer>& info,
    const uint* indices,
    size_t n,
    const std::shared_ptr<BackendRuntime>& runtime,
    const BackendQueue* queue) {
  const auto effective_queue = make_effective_queue(runtime, queue);
  detail::throw_if_status_error(
      runtime->synchronize_queue(*effective_queue),
      "reorder_densification_info sync");
  if (!info || n == 0) {
    return nullptr;
  }
  CHECK_THROW(indices != nullptr);
  auto new_info = create_device_buffer_for<DensificationInfo>(*runtime, n, "densification_reorder");
  const DensificationInfo* old_info_ptr = buffer_data_const<DensificationInfo>(info);
  DensificationInfo* new_info_ptr = buffer_data<DensificationInfo>(new_info);
  for (size_t i = 0; i < n; ++i) {
    new_info_ptr[i] = old_info_ptr[indices[i]];
  }
  return new_info;
}

}  // namespace tinygs
