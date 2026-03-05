#pragma once

#include <memory>

#include "tinygs/platform/backend_types.hpp"
#include "tinygs/platform/runtime_contract.hpp"

namespace tinygs {

Result<BackendRuntime> create_backend_runtime(const BackendConfig& config);

}  // namespace tinygs
