#pragma once

#include <memory>

#include "tinygs/platform/backend_context.hpp"

namespace tinygs {

std::shared_ptr<BackendContext> create_backend_context(const BackendConfig& config);

}  // namespace tinygs
