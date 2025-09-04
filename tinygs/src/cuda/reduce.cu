#include "tinygs/cuda/reduce.hpp"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

namespace tinygs {

float gpu_sum(float *data, int size) {
  return thrust::reduce(thrust::device_ptr<float>(data),
                        thrust::device_ptr<float>(data + size), 0.0f,
                        thrust::plus<float>());
}

float mean(float *data, int size) { return gpu_sum(data, size) / size; }

} // namespace tinygs