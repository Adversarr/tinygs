#pragma once

#include <vector>
#include <string>
#include <tinygs/cuda/common_host.hpp>

namespace tinygs {

/// @brief Point cloud with color.
struct PointCloud {
  std::vector<vec3> points;
  std::vector<vec3> colors;
};

/// @brief Load point cloud from COLMAP pointcloud file.
PointCloud load_from_colmap_file(const std::string &filename);

/// @brief Load point cloud from PLY file.
PointCloud load_ply(const std::string& filename);

}  // namespace tinygs