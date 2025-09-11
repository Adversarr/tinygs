#pragma once

#include "tinygs/core/pointcloud.hpp"
#include "tinygs/core/gaussian.hpp"

namespace tinygs {

class InitializationBase {
public:
  InitializationBase() = default;
  virtual ~InitializationBase() = default;
  virtual void initialize(const PointCloud& pointcloud) = 0;

  const Gaussian3d& gaussians() const { return m_gaussians; }

  virtual void set_params(const json& params) = 0;
  virtual json get_params() const = 0;

protected:
  Gaussian3d m_gaussians;
};

} // namespace tinygs