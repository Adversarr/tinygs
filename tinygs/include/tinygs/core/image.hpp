#pragma once
#include "tinygs/common.hpp"

namespace tinygs {

enum class ImageFormat {
  HWC,
  CHW,
};

template <typename T> struct Image {
  uint16_t width, height;
  uint16_t channels; // 3, 4 is supported.
  ImageFormat format;
  PitchedPtr<T> data;
};

} // namespace tinygs