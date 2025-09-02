#include "tinygs/core/pointcloud.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace tinygs {

// RGB normalization factor to convert from [0,255] to [0,1]
static constexpr float RGB_NORMALIZATION_FACTOR = 255.0f;

PointCloud load_from_colmap_file(const std::string &filename) {
  TINYGS_TIMER("load_from_colmap_file");
  PointCloud pointcloud;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return pointcloud;
  }

  std::string line;
  std::vector<vec3> points;
  std::vector<vec3> colors;

  // Skip header lines that start with '#'
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream iss(line);
    std::vector<float> values;
    float value;

    // Read all values from the line
    while (iss >> value) {
      values.push_back(value);
    }

    // COLMAP points3D.txt format:
    // POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] ...
    // We need columns 1,2,3 (X,Y,Z) and 4,5,6 (R,G,B)
    if (values.size() >= 7) {
      // Extract XYZ coordinates (columns 1,2,3)
      vec3 point(values[1], values[2], values[3]);
      points.push_back(point);

      // Extract RGB colors (columns 4,5,6) and normalize to [0,1]
      vec3 color(values[4] / RGB_NORMALIZATION_FACTOR,
                 values[5] / RGB_NORMALIZATION_FACTOR,
                 values[6] / RGB_NORMALIZATION_FACTOR);
      colors.push_back(color);
    }
  }

  file.close();

  // Move the vectors to the pointcloud struct
  pointcloud.points = std::move(points);
  pointcloud.colors = std::move(colors);

  // // Calculate mean of points
  // vec3 point_mean = vec3(0.0f);
  // for (const auto& p : pointcloud.points) {
  //   point_mean += p;
  // }
  // point_mean /= static_cast<float>(pointcloud.points.size());

  // // Calculate standard deviation of points
  // vec3 point_std = vec3(0.0f);
  // for (const auto& p : pointcloud.points) {
  //   vec3 diff = p - point_mean;
  //   point_std += vec3(diff.x * diff.x, diff.y * diff.y, diff.z * diff.z);
  // }
  // point_std = vec3(
  //   std::sqrt(point_std.x / pointcloud.points.size()),
  //   std::sqrt(point_std.y / pointcloud.points.size()),
  //   std::sqrt(point_std.z / pointcloud.points.size())
  // );

  // // Calculate mean of colors
  // vec3 color_mean = vec3(0.0f);
  // for (const auto& c : pointcloud.colors) {
  //   color_mean += c;
  // }
  // color_mean /= static_cast<float>(pointcloud.colors.size());

  // // Calculate standard deviation of colors
  // vec3 color_std = vec3(0.0f);
  // for (const auto& c : pointcloud.colors) {
  //   vec3 diff = c - color_mean;
  //   color_std += vec3(diff.x * diff.x, diff.y * diff.y, diff.z * diff.z);
  // }
  // color_std = vec3(
  //   std::sqrt(color_std.x / pointcloud.colors.size()),
  //   std::sqrt(color_std.y / pointcloud.colors.size()),
  //   std::sqrt(color_std.z / pointcloud.colors.size())
  // );

  // log_info("Points mean={}, std={}", tinygs::to_string(point_mean), tinygs::to_string(point_std));
  // log_info("Colors mean={}, std={}", tinygs::to_string(color_mean), tinygs::to_string(color_std));

  log_info("Loaded {} 3D points.", pointcloud.points.size());
  return pointcloud;
}

}  // namespace tinygs
