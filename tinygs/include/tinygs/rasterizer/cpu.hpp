#pragma once
// CPU reference rasterizer implementation.
// Naive but strictly correct according to docs/KHR_gaussian_splatting.md and docs/MATH.md.
// Used for testing and as a ground-truth reference for GPU implementations.

#include "tinygs/rasterizer/rasterizer.hpp"

namespace tinygs {

/// @brief CPU reference rasterizer: naive, correct, and deterministic.
///
/// Implements the full forward and backward pass of 3D Gaussian Splatting
/// entirely on the CPU. Data is copied from GPU to host, processed, and
/// results are copied back. This is intentionally slow but serves as the
/// ground-truth reference for validating GPU rasterizer implementations.
///
/// The implementation follows the KHR_gaussian_splatting specification and
/// the MATH.md documentation exactly:
///   - Covariance construction from quaternion + scale
///   - EWA projection to 2D
///   - Tile-based Gaussian-to-pixel assignment (3-sigma cutoff)
///   - Front-to-back alpha compositing
///   - Full SH evaluation up to degree 3 with Condon-Shortley phase
///   - Backward pass with analytic gradients for all parameters
class CPUReferenceRasterizer : public RasterizerBase {
public:
  CPUReferenceRasterizer() = default;
  ~CPUReferenceRasterizer() override = default;

  void forward(const RasterizeContext& ctx) override;
  void backward(RasterizeContext& ctx) override;

  json get_params() const override;
  void set_params(const json& j) override;
};

}  // namespace tinygs