#include "tinygs/utils/inspect_change.hpp"
#include "tinygs/cuda/vec.hpp"
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cmath>

template <typename T>
float diff(
  const T* a,
  const T* b,
  int size
) {
  auto diff = thrust::transform_reduce(
      thrust::device,
      thrust::make_zip_iterator(thrust::make_tuple(a, b)),
      thrust::make_zip_iterator(thrust::make_tuple(a + size, b + size)),
      [] __device__(const thrust::tuple<T, T> &t) -> float {
        auto diff = glm::length(thrust::get<0>(t) - thrust::get<1>(t));
        return diff * diff;
      },
      0.f, thrust::plus<float>{});

  return std::sqrt(diff / size);
}

// Explicit template instantiations
template float diff<tinygs::vec3>(const tinygs::vec3*, const tinygs::vec3*, int);
template float diff<tinygs::vec4>(const tinygs::vec4*, const tinygs::vec4*, int);
template float diff<float>(const float*, const float*, int);

namespace tinygs {

void InspectChange::step() {
  // inspect the difference of buffers.
  const auto& current = m_gaussian;
  const auto& old = m_gaussian_old;

  if (current->size() != old->size()) {
    printf("Size changed: %zu -> %zu\n", old->size(), current->size());
  } else {
    // Compare means
    float means_diff = diff(
      thrust::raw_pointer_cast(current->means().data()),
      thrust::raw_pointer_cast(old->means().data()),
      current->means().size()
    );
    
    // Compare opacities
    float opacities_diff = diff(
      thrust::raw_pointer_cast(current->opacities().data()),
      thrust::raw_pointer_cast(old->opacities().data()),
      current->opacities().size()
    );
    
    // Compare rotations
    float rotations_diff = diff(
      thrust::raw_pointer_cast(current->rotations().data()),
      thrust::raw_pointer_cast(old->rotations().data()),
      current->rotations().size()
    );
    
    // Compare scales
    float scales_diff = diff(
      thrust::raw_pointer_cast(current->scales().data()),
      thrust::raw_pointer_cast(old->scales().data()),
      current->scales().size()
    );
    
    // Compare sh_coefficient_0
    float sh0_diff = diff(
      thrust::raw_pointer_cast(current->sh_coefficient_0().data()),
      thrust::raw_pointer_cast(old->sh_coefficient_0().data()),
      current->sh_coefficient_0().size()
    );
    
    // Compare sh_coefficients_rest
    float sh_rest_diff = diff(
      thrust::raw_pointer_cast(current->sh_coefficients_rest().data()),
      thrust::raw_pointer_cast(old->sh_coefficients_rest().data()),
      current->sh_coefficients_rest().size()
    );
    
    printf("Gaussian changes - means: %.6f, opacities: %.6f, rotations: %.6f, scales: %.6f, sh0: %.6f, sh_rest: %.6f\n",
           means_diff, opacities_diff, rotations_diff, scales_diff, sh0_diff, sh_rest_diff);
  }
  
  // Update the old copy with current state
  m_gaussian_old = current->clone();
}

}  // namespace tinygs
