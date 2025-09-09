#include "tinygs/utils/image_format.hpp"
#include <stdexcept>

namespace tinygs {

#define CHW(c, h, w) (c * height * width + h * width + w)
#define HWC(h, w, c) (h * width * channels + w * channels + c)

void hwc_to_chw(const float* src, float* dst, const ImageShape& shape) {
    const int width = shape.width;
    const int height = shape.height;
    const int channels = shape.channel;
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                dst[CHW(c, h, w)] = src[HWC(h, w, c)];
            }
        }
    }
}

void chw_to_hwc(const float* src, float* dst, const ImageShape& shape) {
    const int width = shape.width;
    const int height = shape.height;
    const int channels = shape.channel;
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                dst[HWC(h, w, c)] = src[CHW(c, h, w)];
            }
        }
    }
}

void hwc_to_chw(const uint8_t* src, uint8_t* dst, const ImageShape& shape) {
    const int width = shape.width;
    const int height = shape.height;
    const int channels = shape.channel;
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                dst[CHW(c, h, w)] = src[HWC(h, w, c)];
            }
        }
    }
}

void chw_to_hwc(const uint8_t* src, uint8_t* dst, const ImageShape& shape) {
    const int width = shape.width;
    const int height = shape.height;
    const int channels = shape.channel;
    
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                dst[HWC(h, w, c)] = src[CHW(c, h, w)];
            }
        }
    }
}

} // namespace tinygs