#pragma once
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

enum class DataType {
  Float32,
  // To make things simple, we only support fp16, instead of bf16 to store the image.
  Float16,
  UInt8,
};

template <>
inline DataType from_string(const std::string& str) {
  if (str == "float32") {
    return DataType::Float32;
  } else if (str == "float16") {
    return DataType::Float16;
  } else if (str == "uint8") {
    return DataType::UInt8;
  } else {
    throw std::runtime_error(fmt::format("Unknown data type: {}", str));
  }
}

inline std::string to_string(const DataType& data_type) {
  switch (data_type) {
    case DataType::Float32: return "float32";
    case DataType::Float16: return "float16";
    case DataType::UInt8: return "uint8";
    default: throw std::runtime_error(fmt::format("Unknown data type: {}", (int)data_type));
  }
}

struct ImageShape {
  uint32_t width = 0, height = 0;
  uint32_t channel = 0; ///< Number of channels (3 or 4 supported)

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
  ImageShape shape{};
  DataType data_type = DataType::Float32;
  void* data = nullptr;

  Image() = default;
  Image(ImageShape shape, DataType data_type, void* data) : shape(shape), data_type(data_type), data(data) {}
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
