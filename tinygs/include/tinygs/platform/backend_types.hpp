#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "tinygs/common.hpp"

struct CUstream_st;

namespace tinygs {

enum class BackendType : uint8_t {
  Cuda = 0,
  Hip = 1,
  Metal = 2,
};

using BackendStream = CUstream_st*;

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
  std::transform(type.begin(), type.end(), type.begin(),
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
