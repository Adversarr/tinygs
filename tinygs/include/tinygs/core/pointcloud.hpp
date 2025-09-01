#pragma once

#include <vector>
#include <string>
#include <tinygs/common.hpp>

namespace tinygs {

/**
 * @brief Simplest point cloud struct.
 * 
 */
struct PointCloud {
  std::vector<vec3> points;
  std::vector<vec3> colors;
};

/**
 * @brief Load point cloud from COLMAP pointcloud file.
 * 
 * @param filename 
 * @return PointCloud 
 */
PointCloud load_from_colmap_file(const std::string &filename);

}  // namespace tinygs