#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "tinygs/common.hpp"

namespace tinygs {

/// Backend selected at build or runtime.
enum class BackendType : uint8_t {
  Cuda = 0,
  Hip = 1,
  Metal = 2,
};

/// Opaque native stream handle passed through backend-neutral APIs.
struct BackendStream {
  void* handle = nullptr;

  constexpr BackendStream() = default;
  constexpr BackendStream(std::nullptr_t) : handle(nullptr) {}
  constexpr BackendStream(void* stream_handle) : handle(stream_handle) {}

  template <typename PointerType, typename = std::enable_if_t<std::is_pointer_v<PointerType>>>
  operator PointerType() const {
    return reinterpret_cast<PointerType>(handle);
  }

  explicit operator bool() const noexcept { return handle != nullptr; }
};

inline std::string to_string(BackendType type) {
  switch (type) {
    case BackendType::Cuda:
      return "cuda";
    case BackendType::Hip:
      return "hip";
    case BackendType::Metal:
      return "metal";
    default:
      throw std::runtime_error("Unknown backend type enum value.");
  }
}

inline BackendType backend_type_from_string(std::string type) {
  std::transform(
      type.begin(),
      type.end(),
      type.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (type == "cuda") {
    return BackendType::Cuda;
  }
  if (type == "hip") {
    return BackendType::Hip;
  }
  if (type == "metal") {
    return BackendType::Metal;
  }
  throw std::runtime_error("Unknown backend type: " + type);
}

/// Runtime backend selection parsed from config.
struct BackendConfig {
  BackendType type = BackendType::Cuda;
  int device = 0;

  void from_json(const json& config) {
    if (!config.contains("type")) {
      throw std::runtime_error("backend.type is required.");
    }
    type = backend_type_from_string(config.at("type").get<std::string>());
    if (config.contains("device")) {
      device = config.at("device").get<int>();
    }
  }

  json to_json() const {
    json config;
    config["type"] = to_string(type);
    config["device"] = device;
    return config;
  }
};

}  // namespace tinygs
