#include "tinygs/platform/backend_factory.hpp"

#include <stdexcept>

namespace tinygs {

std::shared_ptr<BackendContext> create_backend_context(const BackendConfig& config) {
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
