#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "tinygs/platform/backend_types.hpp"

namespace tinygs {

enum class BackendErrorCode : uint8_t {
  Success = 0,
  InvalidArgument = 1,
  Unsupported = 2,
  OutOfMemory = 3,
  Timeout = 4,
  DeviceLost = 5,
  SynchronizationError = 6,
  RuntimeFailure = 7,
  UnknownFailure = 8,
};

inline const char* to_string(BackendErrorCode code) noexcept {
  switch (code) {
    case BackendErrorCode::Success:
      return "success";
    case BackendErrorCode::InvalidArgument:
      return "invalid_argument";
    case BackendErrorCode::Unsupported:
      return "unsupported";
    case BackendErrorCode::OutOfMemory:
      return "out_of_memory";
    case BackendErrorCode::Timeout:
      return "timeout";
    case BackendErrorCode::DeviceLost:
      return "device_lost";
    case BackendErrorCode::SynchronizationError:
      return "synchronization_error";
    case BackendErrorCode::RuntimeFailure:
      return "runtime_failure";
    case BackendErrorCode::UnknownFailure:
      return "unknown_failure";
    default:
      return "unknown_backend_error_code";
  }
}

struct BackendError {
  BackendErrorCode code = BackendErrorCode::Success;
  BackendType backend = BackendType::Cuda;
  std::string operation;
  std::string message;

  bool ok() const noexcept { return code == BackendErrorCode::Success; }
};

inline BackendError backend_success(BackendType backend, std::string operation = {}) {
  BackendError status;
  status.code = BackendErrorCode::Success;
  status.backend = backend;
  status.operation = std::move(operation);
  return status;
}

inline BackendError backend_error(BackendType backend,
                                  BackendErrorCode code,
                                  std::string operation,
                                  std::string message = {}) {
  BackendError status;
  status.code = code;
  status.backend = backend;
  status.operation = std::move(operation);
  status.message = std::move(message);
  return status;
}

inline std::string to_string(const BackendError& status) {
  std::string out = "backend=" + to_string(status.backend) +
                    " code=" + std::string(to_string(status.code));
  if (!status.operation.empty()) {
    out += " op=" + status.operation;
  }
  if (!status.message.empty()) {
    out += " message=" + status.message;
  }
  return out;
}

}  // namespace tinygs
