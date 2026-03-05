#include "tinygs/platform/backend_factory.hpp"

#include <stdexcept>

#include "tinygs/platform/backend_build.hpp"

namespace tinygs {

std::shared_ptr<BackendContext> create_backend_context(const BackendConfig& config) {
  if (config.type != compiled_backend_type()) {
    throw std::runtime_error(
        "Requested backend '" + to_string(config.type) +
        "' does not match compiled backend '" + std::string(compiled_backend_name()) + "'.");
  }

  switch (config.type) {
    case BackendType::Cuda:
      return std::make_shared<CudaBackendContext>(config.device);
    case BackendType::Hip:
      throw std::runtime_error("HIP backend is planned but not implemented yet.");
    case BackendType::Metal:
      throw std::runtime_error("Metal backend is planned but not implemented yet.");
    default:
      throw std::runtime_error("Unknown backend type.");
  }
}

}  // namespace tinygs
