#include "tinygs/cuda/stat.hpp"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <glm/gtx/string_cast.hpp>

namespace tinygs {

template<typename T>
BufferStat<T> compute_buffer_stat_gpu(const T* data, size_t size) {
  BufferStat<T> stat;

  // mean
  stat.mean = thrust::reduce(
    thrust::device,
    thrust::device_ptr<T>(const_cast<T*>(data)),
    thrust::device_ptr<T>(const_cast<T*>(data)) + size,
    T(0),
    thrust::plus<T>()
  ) / (float) size;

  // std
  stat.std = thrust::transform_reduce(
    thrust::device,
    thrust::device_ptr<T>(const_cast<T*>(data)),
    thrust::device_ptr<T>(const_cast<T*>(data)) + size,
    [mean=stat.mean] __device__ (T x) -> T { return (x - mean) * (x - mean); },
    T(0),
    thrust::plus<T>()
  ) / (float) size;
  stat.std = sqrt(stat.std);

  // avg norm2
  stat.avg_norm2 = thrust::transform_reduce(
    thrust::device,
    thrust::device_ptr<T>(const_cast<T*>(data)),
    thrust::device_ptr<T>(const_cast<T*>(data)) + size,
    [] __device__ (T x) -> float { return glm::length(x); },
    0.f,
    thrust::plus<float>()
  ) / (float) size;

  return stat;
}

// Instantiate
template BufferStat<float> compute_buffer_stat_gpu(const float *data, size_t size);
template BufferStat<vec2> compute_buffer_stat_gpu(const vec2 *data, size_t size);
template BufferStat<vec3> compute_buffer_stat_gpu(const vec3 *data, size_t size);
template BufferStat<vec4> compute_buffer_stat_gpu(const vec4 *data, size_t size);

template<typename T>
std::string to_string(const BufferStat<T>& stat) {
  return fmt::format("mean: {}, std: {}, avg_norm2: {}", 
      glm::to_string(stat.mean), glm::to_string(stat.std), stat.avg_norm2);
}

template<>
std::string to_string(const BufferStat<float>& stat) {
  return fmt::format("mean: {}, std: {}, avg_norm2: {}", stat.mean, stat.std, stat.avg_norm2);
}

template std::string to_string(const BufferStat<vec2>& stat);
template std::string to_string(const BufferStat<vec3>& stat);
template std::string to_string(const BufferStat<vec4>& stat);

}
