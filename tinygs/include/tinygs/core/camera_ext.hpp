#pragma once
#include "tinygs/core/camera.hpp"

namespace tinygs {

enum class InterpolationMethod: int {
  Linear,
  MaxInterpolationMethod,
};


/// @brief Interpolate between two camera extrinsics
/// @param a First camera extrinsics
/// @param b Second camera extrinsics
/// @param t Interpolation parameter in [0, 1]
/// @param method Interpolation method
/// @return Interpolated camera extrinsics
CameraExtrinsics interpolate(
  const CameraExtrinsics& a,
  const CameraExtrinsics& b,
  float t,
  InterpolationMethod method = InterpolationMethod::Linear);

}