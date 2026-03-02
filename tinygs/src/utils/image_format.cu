#include "tinygs/utils/image_format.hpp"
#include "tinygs/common.hpp"
#include "tinygs/cuda/common_device.cuh"
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

// Convert standard cv2 image (HWC, BGR) to our CHW+Tiled format (RGB)
void from_cv2(uint8_t* dst, const uint8_t* src, const ImageShape& shape) {
    const uint32_t width = shape.width;
    const uint32_t height = shape.height;
    const uint32_t channels = shape.channel;

    if (channels != 3) {
        throw std::runtime_error("from_cv2 expects 3 channels (RGB)");
    }

    const uint32_t tiled_w = shape.tiled_width();
    const uint32_t total_pix = shape.padded_width() * shape.padded_height();

    for (uint32_t c = 0; c < channels; ++c) {
        for (uint32_t h = 0; h < height; ++h) {
            for (uint32_t w = 0; w < width; ++w) {
                const uint32_t lin = get_linear_index_tiled(h, w, tiled_w);
                // src is HWC in BGR order; map to RGB
                const uint32_t src_idx = h * width * channels + w * channels + (2 - c);
                dst[c * total_pix + lin] = src[src_idx];
            }
        }
    }
}

// Convert our CHW+Tiled format (RGB) to standard cv2 image (HWC, BGR)
void to_cv2(uint8_t* dst, const uint8_t* src, const ImageShape& shape) {
    const uint32_t width = shape.width;
    const uint32_t height = shape.height;
    const uint32_t channels = shape.channel;

    if (channels != 3) {
        throw std::runtime_error("to_cv2 expects 3 channels (RGB)");
    }

    const uint32_t tiled_w = shape.tiled_width();
    const uint32_t total_pix = shape.padded_width() * shape.padded_height();

    for (uint32_t h = 0; h < height; ++h) {
        for (uint32_t w = 0; w < width; ++w) {
            const uint32_t lin = get_linear_index_tiled(h, w, tiled_w);
            for (uint32_t c = 0; c < channels; ++c) {
                const uint8_t val = src[c * total_pix + lin]; // RGB
                const uint32_t dst_idx = h * width * channels + w * channels + (2 - c); // BGR
                dst[dst_idx] = val;
            }
        }
    }
}

__global__ void half_to_float_kernel(int n, const float16_t *src, float *dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx] = __half2float(src[idx]);
}

__global__ void float_to_half_kernel(int n, const float *src, float16_t *dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx] = __float2half(src[idx]);
}

void half_to_float_gpu(float* dst, const float16_t* src, int n, cudaStream_t stream) {
    linear_kernel(half_to_float_kernel, 0, stream, n, src, dst);
}

void float_to_half_gpu(float16_t* dst, const float* src, int n, cudaStream_t stream) {
    linear_kernel(float_to_half_kernel, 0, stream, n, src, dst);
}

} // namespace tinygs