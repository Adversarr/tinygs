#pragma once

#include "tinygs/core/image.hpp"
#include <algorithm>
#include <cstdint>

namespace tinygs {

/**
 * @brief Convert image data between HWC and CHW formats
 * 
 * This utility provides functions to convert image data between Height-Width-Channel (HWC)
 * and Channel-Height-Width (CHW) formats, assuming the buffer is on CPU.
 */

/**
 * @brief Convert image data from HWC format to CHW format
 * 
 * @param src Source buffer in HWC format (height * width * channels)
 * @param dst Destination buffer in CHW format (channels * height * width)
 * @param shape Image shape containing width, height, and channels
 */
void hwc_to_chw(const float* src, float* dst, const ImageShape& shape);

/**
 * @brief Convert image data from CHW format to HWC format
 * 
 * @param src Source buffer in CHW format (channels * height * width)
 * @param dst Destination buffer in HWC format (height * width * channels)
 * @param shape Image shape containing width, height, and channels
 */
void chw_to_hwc(const float* src, float* dst, const ImageShape& shape);

/**
 * @brief Convert image data from HWC format to CHW format (uint8 version)
 * 
 * @param src Source buffer in HWC format (height * width * channels)
 * @param dst Destination buffer in CHW format (channels * height * width)
 * @param shape Image shape containing width, height, and channels
 */
void hwc_to_chw(const uint8_t* src, uint8_t* dst, const ImageShape& shape);

/**
 * @brief Convert image data from CHW format to HWC format (uint8 version)
 * 
 * @param src Source buffer in CHW format (channels * height * width)
 * @param dst Destination buffer in HWC format (height * width * channels)
 * @param shape Image shape containing width, height, and channels
 */
void chw_to_hwc(const uint8_t* src, uint8_t* dst, const ImageShape& shape);

} // namespace tinygs