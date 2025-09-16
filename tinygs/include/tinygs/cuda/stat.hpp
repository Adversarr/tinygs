#pragma once
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

/// @brief Statistical information for buffer data
template<typename T>
struct BufferStat {
  T mean;     ///< Mean value
  T std;      ///< Standard deviation

  float avg_norm2 = 0.0f;  ///< Average L2 norm
};

/// @brief Compute buffer statistics on GPU
template<typename T>
BufferStat<T> compute_buffer_stat_gpu(const T* data, size_t size);

/// @brief Convert buffer statistics to string
template<typename T>
std::string to_string(const BufferStat<T>& stat);

}