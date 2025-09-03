#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "cuda/gpu_memory.hpp"
#include "tinygs/loss/loss.hpp"

namespace tinygs {

float sum_loss(const LossContext& ctx) {
  GPUBuffer<float> buf(cudaStream_t(nullptr), 1);

  uint32_t total = ctx.loss.size();
  const float* data = ctx.loss.data;

  float loss = thrust::reduce(thrust::device_ptr<const float>(data), thrust::device_ptr<const float>(data + total), 0.0f,
                              thrust::plus<float>());
  return loss;
}
}  // namespace tinygs
