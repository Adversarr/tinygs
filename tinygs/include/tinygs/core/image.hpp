#pragma once
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

enum class ImageFormat {
  HWC,
  CHW,
};

struct ImageShape {
  uint16_t width, height;
  uint16_t channels; // 3, 4 is supported.

  TINYGS_HOST_DEVICE bool operator==(const ImageShape& other) const noexcept {
    return width == other.width && height == other.height && channels == other.channels;
  }

  TINYGS_HOST_DEVICE bool operator!=(const ImageShape& other) const noexcept {
    return !(*this == other);
  }
};

inline std::string to_string(const ImageShape& shape) {
  return fmt::format("ImageShape{{width={}, height={}, channels={}}}", shape.width, shape.height, shape.channels);
}

inline std::string to_string(const ImageFormat& format) {
  switch (format) {
    case ImageFormat::HWC:
      return "HWC";
    case ImageFormat::CHW:
      return "CHW";
  }
  return "Unknown";
}

template <typename T> struct Image {
  ImageShape shape;
  ImageFormat format = ImageFormat::HWC;
  T* data;

  TINYGS_HOST_DEVICE explicit operator bool() const noexcept {
    return data != nullptr;
  }
};

} // namespace tinygs