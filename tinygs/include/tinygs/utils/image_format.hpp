#pragma once

#include "tinygs/core/image.hpp"
#include <algorithm>
#include <cstdint>

namespace tinygs {

/// @brief Convert image data from HWC format to CHW format
void hwc_to_chw(const float* src, float* dst, const ImageShape& shape);

/// @brief Convert image data from CHW format to HWC format
void chw_to_hwc(const float* src, float* dst, const ImageShape& shape);

/// @brief Convert image data from HWC format to CHW format (uint8 version)
void hwc_to_chw(const uint8_t* src, uint8_t* dst, const ImageShape& shape);

/// @brief Convert image data from CHW format to HWC format (uint8 version)
void chw_to_hwc(const uint8_t* src, uint8_t* dst, const ImageShape& shape);

} // namespace tinygs