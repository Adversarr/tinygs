#pragma once

#include <vector>
#include <string>
#include <tinygs/cuda/common_host.hpp>
#include <tinygs/core/gaussian.hpp>

namespace tinygs {

/// @brief Point cloud with color.
struct PointCloud {
  std::vector<vec3> points;
  std::vector<vec3> colors;
};

/// @brief Load point cloud from COLMAP pointcloud file.
PointCloud load_from_colmap(const std::string &filename);

/// @brief Load point cloud from PLY file.
PointCloud load_ply(const std::string& filename);

/// @brief Load point cloud from file, supports COLMAP and PLY formats based on file extension.
PointCloud load_point_cloud(const std::string& filename);

/// @brief Save Gaussian3d data to PLY file.
void save_ply(const std::string& filename, const Gaussian3d& gs);

}  // namespace tinygs