#include "tinygs/cuda/reduce.hpp"
#include "tinygs/cuda/common_host.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include <cfloat>

namespace tinygs {

float gpu_sum(float *data, int size, const BackendQueue* queue) {
  cudaStream_t cuda_stream = to_cuda_stream(queue);
  return thrust::reduce(thrust::cuda::par.on(cuda_stream),
                        thrust::device_ptr<float>(data),
                        thrust::device_ptr<float>(data + size), 0.0f,
                        thrust::plus<float>());
}

void gpu_mean_vec3(const vec3* data, int size, vec3& out, const BackendQueue* queue) {
  if (size <= 0) {
    out = vec3(0.0f, 0.0f, 0.0f);
    return;
  }
  cudaStream_t cuda_stream = to_cuda_stream(queue);
  out = thrust::transform_reduce(
    thrust::cuda::par.on(cuda_stream),
    thrust::device_ptr<const vec3>(data),
    thrust::device_ptr<const vec3>(data + size),
    [inv_s = 1.0f / static_cast<float>(size)] __device__ (const vec3& p) -> vec3 {
      return p * inv_s;
    },
    vec3(0.0f, 0.0f, 0.0f),
    thrust::plus<vec3>()
  );
}

}