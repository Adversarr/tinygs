#pragma once
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

enum class ImageDataType {
  Float32,
  UInt8,
};

struct ImageShape {
  uint32_t width, height;
  uint32_t channel; ///< Number of channels (3 or 4 supported)

  TINYGS_HOST_DEVICE bool operator==(const ImageShape& other) const noexcept {
    return width == other.width && height == other.height && channel == other.channel;
  }

  TINYGS_HOST_DEVICE bool operator!=(const ImageShape& other) const noexcept {
    return !(*this == other);
  }

  TINYGS_HOST_DEVICE uint32_t size() const noexcept {
    return width * height * channel;
  }

  TINYGS_HOST_DEVICE uint32_t tiled_width() const noexcept { return (width + kImageTileMask) >> kImageTileLog2; }

  TINYGS_HOST_DEVICE uint32_t tiled_height() const noexcept { return (height + kImageTileMask) >> kImageTileLog2; }

  TINYGS_HOST_DEVICE uint32_t padded_width() const noexcept { return tiled_width() << kImageTileLog2; }

  TINYGS_HOST_DEVICE uint32_t padded_height() const noexcept { return tiled_height() << kImageTileLog2; }

  TINYGS_HOST_DEVICE uint32_t padded_size() const noexcept { return padded_width() * padded_height() * channel; }
};

inline std::string to_string(const ImageShape& shape) {
  return fmt::format("ImageShape{{width={}, height={}, channels={}}}", shape.width, shape.height, shape.channel);
}

struct Image {
  ImageShape shape;
  ImageDataType data_type = ImageDataType::Float32;
  void* data;

  Image() = default;
  Image(ImageShape shape, ImageDataType data_type, void* data) : shape(shape), data_type(data_type), data(data) {}
  Image(const Image& other) = default;
  Image(Image&& other) = default;
  Image& operator=(const Image& other) = default;
  Image& operator=(Image&& other) = default;

  TINYGS_HOST_DEVICE explicit operator bool() const noexcept {
    return data != nullptr;
  }

  TINYGS_HOST_DEVICE uint32_t size() const noexcept {
    return shape.size();
  }
};

} // namespace tinygs
