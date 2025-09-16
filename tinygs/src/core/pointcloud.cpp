#include "tinygs/core/pointcloud.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "./happly.h"
#include "tinygs/utils/scope_timer.hpp"

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


  log_info("Loaded {} 3D points.", pointcloud.points.size());
  return pointcloud;
}


PointCloud load_ply(std::string filename) {
  happly::PLYData ply_in(filename);
  auto points = ply_in.getVertexPositions();
  auto colors = ply_in.getVertexColors();
  log_info("Loaded PLY file {} with {} points and {} colors", filename, points.size(), colors.size());
  PointCloud pc;
  pc.points.reserve(points.size());
  pc.colors.reserve(colors.size());

  for (int i = 0; i < points.size(); i++) {
    pc.points.push_back({(float)points[i][0], (float)points[i][1], (float)points[i][2]});
    pc.colors.push_back({
      (float) colors[i][0] / RGB_NORMALIZATION_FACTOR,
      (float) colors[i][1] / RGB_NORMALIZATION_FACTOR,
      (float) colors[i][2] / RGB_NORMALIZATION_FACTOR});
  }
  return pc;
}


}  // namespace tinygs
