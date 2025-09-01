#pragma once

#include "tinygs/core/pointcloud.hpp"
#include "tinygs/core/gaussian.hpp"

namespace tinygs {

class InitializationBase {
public:
  InitializationBase() = default;
  virtual ~InitializationBase() = default;
  virtual void initialize(PointCloud& pointcloud) = 0;

  const Gaussian3d& gaussians() const { return m_gaussians; }

protected:
  Gaussian3d m_gaussians;
};

} // namespace tinygs