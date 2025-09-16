#pragma once
namespace tinygs {

/// @brief Compute sum of array elements on GPU
float gpu_sum(float* data, int size);

/// @brief Compute mean of array elements
float mean(float* data, int size);

}