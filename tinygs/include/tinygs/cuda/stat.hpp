#pragma once
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

template<typename T>
struct BufferStat {
  T mean;
  T std;

  float avg_norm2 = 0.0f;
};

template<typename T>
BufferStat<T> compute_buffer_stat_gpu(const T* data, size_t size);

template<typename T>
std::string to_string(const BufferStat<T>& stat);

}