#pragma once

#include "tinygs/common.hpp"
#include "tinygs/platform/backend_types.hpp"

namespace tinygs {

float gpu_sum(float* data, int size, BackendStream stream = nullptr);

void gpu_mean_vec3(const vec3* data, int size, vec3& out, BackendStream stream = nullptr);

}