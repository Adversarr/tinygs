#pragma once

#include <memory>

#include "tinygs/platform/backend_types.hpp"
#include "tinygs/platform/runtime.hpp"

namespace tinygs {

/// Create the runtime for the requested backend and device.
Result<BackendRuntime> create_backend_runtime(const BackendConfig& config);

}  // namespace tinygs
