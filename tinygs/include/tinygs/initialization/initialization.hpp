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

/**
 * @brief Create an initialization object
 *
 * @param initialization_type The type of initialization to create ("knn", "random", etc.)
 * @return std::unique_ptr<InitializationBase> The created initialization
 */
std::unique_ptr<InitializationBase> create_initialization(const std::string& initialization_type);

} // namespace tinygs