#pragma once

#include "tinygs/common.hpp"

namespace tinygs {

class BackendQueue;

float gpu_sum(float* data, int size, const BackendQueue* queue = nullptr);

void gpu_mean_vec3(const vec3* data, int size, vec3& out, const BackendQueue* queue = nullptr);

}