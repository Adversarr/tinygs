#include "tinygs/random/multinomial.hpp"

#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <numeric>
#include <algorithm>
#include "tinygs/cuda/common_host.hpp"
#include "tinygs/random/pcg32.hpp"

namespace tinygs{

namespace {
/// Wraps a thrust::device_vector<int> as a BackendBuffer so the public API stays
/// backend-agnostic while keeping the underlying memory managed by thrust.
class ThrustIntBuffer final : public BackendBuffer {
public:
  explicit ThrustIntBuffer(thrust::device_vector<int>&& vec) : m_vec(std::move(vec)) {}
  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return 0; }
  size_t size_bytes() const noexcept override { return m_vec.size() * sizeof(int); }
  const BufferDesc& desc() const noexcept override { static BufferDesc d; return d; }
  void* data() const noexcept override {
    return const_cast<int*>(thrust::raw_pointer_cast(m_vec.data()));
  }
  void* native_handle() const noexcept override { return data(); }
private:
  thrust::device_vector<int> m_vec;
};

inline std::shared_ptr<BackendBuffer> wrap_thrust_int(thrust::device_vector<int>&& v) {
  return std::make_shared<ThrustIntBuffer>(std::move(v));
}
} // anonymous namespace

__device__ __forceinline__ int lower_bound_cdf(const float* cdf, int K, float u) {
    int lo = 0, hi = K - 1;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (u <= cdf[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

__global__ void multinomial_sample_kernel(
    const float* __restrict__ cdf,
    int K,
    float total_sum,
    int num_samples,
    int seed,
    int* __restrict__ out_indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);

    float r = curand_uniform(&state);
    float u = r * total_sum;
    if (u >= total_sum) {
        u = nextafterf(total_sum, 0.0f);
    }

    int idx = lower_bound_cdf(cdf, K, u);
    out_indices[i] = idx;
}

std::shared_ptr<BackendBuffer> multinomial_cuda_with_replacement(
  const float* d_weights, 
  int K, 
  int num_samples, 
  int seed,
  const BackendQueue* queue)
{
  const cudaStream_t cuda_stream = to_cuda_stream(queue);
  if (K <= 0 || num_samples <= 0) {
      throw std::runtime_error(fmt::format("Invalid K={} or num_samples={}", K, num_samples));
  }

  thrust::device_vector<float> b_cdf(K);
  float* d_cdf = thrust::raw_pointer_cast(b_cdf.data());
  CUDA_CHECK_THROW(cudaMemcpyAsync(d_cdf, d_weights, sizeof(float) * K, cudaMemcpyDeviceToDevice, cuda_stream));

  void* d_temp = nullptr;
  size_t temp_bytes = 0;
  CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_cdf, d_cdf, K, cuda_stream));
  thrust::device_vector<uint8_t> b_temp(temp_bytes);
  d_temp = thrust::raw_pointer_cast(b_temp.data());
  CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_cdf, d_cdf, K, cuda_stream));

  float h_total = 0.0f;
  CUDA_CHECK_THROW(cudaMemcpyAsync(&h_total, d_cdf + (K - 1), sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));

  if (!(h_total > 0.0f) || !isfinite(h_total)) {
      throw std::runtime_error(fmt::format("Invalid weights: h_total={:.4e}", h_total));
  }

  thrust::device_vector<int> b_out(num_samples);
  int* d_out = thrust::raw_pointer_cast(b_out.data());

  int threads = 256;
  int blocks = (num_samples + threads - 1) / threads;
  multinomial_sample_kernel<<<blocks, threads, 0, cuda_stream>>>(d_cdf, K, h_total, num_samples, seed, d_out);
  CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));
  return wrap_thrust_int(std::move(b_out));
}

// CPU implementation of lower_bound_cdf
int lower_bound_cdf_cpu(const float *cdf, int K, float u) {
  int lo = 0, hi = K - 1;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    if (u <= cdf[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

std::vector<int> multinomial_cpu_with_replacement(
  const float* weights,
  int K,
  int num_samples,
  int seed)
{
  if (K <= 0 || num_samples <= 0) {
    throw std::runtime_error(
        fmt::format("Invalid K={} or num_samples={}", K, num_samples));
  }

  // Compute CDF on CPU
  std::vector<float> cdf(K);
  std::partial_sum(weights, weights + K, cdf.begin());

  float total_sum = cdf[K - 1];

  if (!(total_sum > 0.0f) || !std::isfinite(total_sum)) {
    throw std::runtime_error(
        fmt::format("Invalid weights: total_sum={:.4e}", total_sum));
  }

  // Initialize random number generator
  pcg32 rng(seed);

  // Sample indices
  std::vector<int> out_indices(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    float r = rng.next_float();
    float u = r * total_sum;
    if (u >= total_sum) {
      u = std::nextafter(total_sum, 0.0f);
    }

    int idx = lower_bound_cdf_cpu(cdf.data(), K, u);
    out_indices[i] = idx;
  }

  return out_indices;
}

std::shared_ptr<BackendBuffer> multinomial_cuda_cpu(
  const float* d_weights,
  int K,
  int num_samples,
  int seed,
  const BackendQueue* queue)
{
  const cudaStream_t cuda_stream = to_cuda_stream(queue);
  std::vector<float> h_weights(K);
  CUDA_CHECK_THROW(cudaMemcpyAsync(h_weights.data(), d_weights, sizeof(float) * K, cudaMemcpyDeviceToHost, cuda_stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));

  auto h_out = multinomial_cpu_with_replacement(h_weights.data(), K, num_samples, seed);
  thrust::device_vector<int> b_out(num_samples);
  CUDA_CHECK_THROW(cudaMemcpyAsync(thrust::raw_pointer_cast(b_out.data()), h_out.data(), sizeof(int) * num_samples, cudaMemcpyHostToDevice, cuda_stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));
  return wrap_thrust_int(std::move(b_out));
}

// CPU sampling without replacement using Efraimidis–Spirakis PPS scheme
std::vector<int> multinomial_cpu_without_replacement(
  const float* weights,
  int K,
  int num_samples,
  int seed)
{
  if (K <= 0 || num_samples <= 0) {
    throw std::runtime_error(
        fmt::format("Invalid K={} or num_samples={}", K, num_samples));
  }

  pcg32 rng(seed);
  struct KeyIdx { float key; int idx; };
  std::vector<KeyIdx> keys;
  keys.reserve(K);

  float total_pos = 0.0f;
  for (int i = 0; i < K; ++i) {
    float w = weights[i];
    if (w > 0.0f && std::isfinite(w)) {
      total_pos += w;
      float u = rng.next_float();
      // Ensure u in (0,1) open interval to avoid edge cases
      if (u <= 0.0f) u = std::nextafter(0.0f, 1.0f);
      if (u >= 1.0f) u = std::nextafter(1.0f, 0.0f);
      float key = std::pow(u, 1.0f / w);
      keys.push_back({key, i});
    }
  }

  if (total_pos <= 0.0f || keys.empty()) {
    throw std::runtime_error("Invalid weights: no positive entries for sampling without replacement");
  }

  if (num_samples > static_cast<int>(keys.size())) {
    num_samples = static_cast<int>(keys.size());
  }

  // Select top-M entries by key
  std::nth_element(keys.begin(), keys.begin() + num_samples, keys.end(),
                   [](const KeyIdx& a, const KeyIdx& b) { return a.key > b.key; });
  std::sort(keys.begin(), keys.begin() + num_samples,
            [](const KeyIdx& a, const KeyIdx& b) { return a.key > b.key; });

  std::vector<int> out(num_samples);
  for (int i = 0; i < num_samples; ++i) out[i] = keys[i].idx;
  return out;
}

std::shared_ptr<BackendBuffer> multinomial_cuda_cpu_without_replacement(
  const float* d_weights,
  int K,
  int num_samples,
  int seed,
  const BackendQueue* queue)
{
  const cudaStream_t cuda_stream = to_cuda_stream(queue);
  if (K <= 0 || num_samples <= 0) {
    throw std::runtime_error(fmt::format("Invalid K={} or num_samples={}", K, num_samples));
  }

  std::vector<float> h_weights(K);
  CUDA_CHECK_THROW(cudaMemcpyAsync(h_weights.data(), d_weights, sizeof(float) * K, cudaMemcpyDeviceToHost, cuda_stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));

  auto h_out = multinomial_cpu_without_replacement(h_weights.data(), K, num_samples, seed);
  const int actual_num_samples = static_cast<int>(h_out.size());
  thrust::device_vector<int> b_out(actual_num_samples);
  CUDA_CHECK_THROW(cudaMemcpyAsync(
      thrust::raw_pointer_cast(b_out.data()),
      h_out.data(),
      sizeof(int) * actual_num_samples,
      cudaMemcpyHostToDevice,
      cuda_stream));
  CUDA_CHECK_THROW(cudaStreamSynchronize(cuda_stream));
  return wrap_thrust_int(std::move(b_out));
}

}
