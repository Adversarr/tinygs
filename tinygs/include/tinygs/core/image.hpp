#pragma once
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

enum class ImageFormat {
  HWC,
  CHW,
};

enum class ImageDataType {
  Float32,
  UInt8,
};

struct ImageShape {
  uint16_t width, height;
  uint16_t channel; // 3, 4 is supported.

  TINYGS_HOST_DEVICE bool operator==(const ImageShape& other) const noexcept {
    return width == other.width && height == other.height && channel == other.channel;
  }

  TINYGS_HOST_DEVICE bool operator!=(const ImageShape& other) const noexcept {
    return !(*this == other);
  }
};

inline std::string to_string(const ImageShape& shape) {
  return fmt::format("ImageShape{{width={}, height={}, channels={}}}", shape.width, shape.height, shape.channel);
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

struct Image {
  ImageShape shape;
  ImageFormat format = ImageFormat::HWC;
  ImageDataType data_type = ImageDataType::Float32;
  void* data;

  Image() = default;
  Image(ImageShape shape, ImageFormat format, ImageDataType data_type, void* data) 
    : shape(shape), format(format), data_type(data_type), data(data) {}
  Image(const Image& other) = default;
  Image(Image&& other) = default;
  Image& operator=(const Image& other) = default;
  Image& operator=(Image&& other) = default;

  TINYGS_HOST_DEVICE explicit operator bool() const noexcept {
    return data != nullptr;
  }

  TINYGS_HOST_DEVICE uint32_t size() const noexcept {
    return shape.width * shape.height * shape.channel;
  }
};

} // namespace tinygs