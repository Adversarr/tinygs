#pragma once

#include "tinygs/platform/backend_types.hpp"

namespace tinygs {

/// Backend compiled into this build.
inline constexpr BackendType compiled_backend_type() noexcept {
#if defined(TINYGS_BACKEND_CUDA)
  return BackendType::Cuda;
#elif defined(TINYGS_BACKEND_HIP)
  return BackendType::Hip;
#elif defined(TINYGS_BACKEND_METAL)
  return BackendType::Metal;
#else
#error "No compiled backend macro defined. Expected TINYGS_BACKEND_{CUDA|HIP|METAL}."
#endif
}

/// Lowercase backend name for the compiled backend.
inline constexpr const char* compiled_backend_name() noexcept {
  switch (compiled_backend_type()) {
    case BackendType::Cuda:
      return "cuda";
    case BackendType::Hip:
      return "hip";
    case BackendType::Metal:
      return "metal";
    default:
      return "unknown";
  }
}

}  // namespace tinygs
