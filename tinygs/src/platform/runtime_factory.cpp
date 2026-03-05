#include "tinygs/platform/runtime_factory.hpp"

#include "tinygs/platform/backend_build.hpp"

namespace tinygs {

Result<BackendRuntime> create_cuda_backend_runtime(int device);

Result<BackendRuntime> create_backend_runtime(const BackendConfig& config) {
  constexpr const char* operation = "create_backend_runtime";

  if (config.type != compiled_backend_type()) {
    return Result<BackendRuntime>::failure(
        backend_error(config.type,
                      BackendErrorCode::InvalidArgument,
                      operation,
                      "Requested backend '" + to_string(config.type) +
                          "' does not match compiled backend '" +
                          std::string(compiled_backend_name()) + "'."));
  }

  switch (config.type) {
    case BackendType::Cuda:
      return create_cuda_backend_runtime(config.device);
    case BackendType::Hip:
      return Result<BackendRuntime>::failure(
          backend_error(config.type,
                        BackendErrorCode::Unsupported,
                        operation,
                        "HIP backend is planned but not implemented yet."));
    case BackendType::Metal:
      return Result<BackendRuntime>::failure(
          backend_error(config.type,
                        BackendErrorCode::Unsupported,
                        operation,
                        "Metal backend is planned but not implemented yet."));
    default:
      return Result<BackendRuntime>::failure(
          backend_error(config.type,
                        BackendErrorCode::InvalidArgument,
                        operation,
                        "Unknown backend type."));
  }
}

}  // namespace tinygs
