#pragma once
#include <tinygs/common.hpp>
#include <tinygs/cuda/vec.hpp>
#include <tinygs/cuda/gpu_memory.hpp>

#define TINYGS_MAX_IMAGE_WIDTH 2048
#define TINYGS_MAX_IMAGE_HEIGHT 2048

namespace tinygs {

template <typename T>
class GPUImage {
public:
  GPUImage(int width, int height);

  ~GPUImage();

  void resize(int width, int height);

private:
  int m_width;
  int m_height;
  GPUMemory<T> m_memory;
};

}
