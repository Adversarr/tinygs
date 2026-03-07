#pragma once

#include "tinygs/core/image.hpp"
#include <algorithm>
#include <cstdint>

namespace tinygs {

class BackendQueue;

/// @brief Convert our CHW+Tiled format (RGB) to standard cv2 image (HWC, BGR)
void to_cv2(uint8_t* dst, const uint8_t* src, const ImageShape& shape);

void half_to_float_gpu(float* dst, const float16_t* src, int n, const BackendQueue* queue = nullptr);
void float_to_half_gpu(float16_t* dst, const float* src, int n, const BackendQueue* queue = nullptr);

} // namespace tinygs
