#include "tinygs/core/pointcloud.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "./happly.h"
#include "tinygs/common.hpp"
#include "tinygs/core/gaussian.hpp"
#include "tinygs/utils/scope_timer.hpp"
#include <nvtx3/nvtx3.hpp>

namespace tinygs {

// RGB normalization factor to convert from [0,255] to [0,1]
static constexpr float RGB_NORMALIZATION_FACTOR = 255.0f;

PointCloud load_from_colmap(const std::string &filename) {
  NVTX3_FUNC_RANGE();
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

PointCloud load_ply(const std::string& filename) {
  happly::PLYData ply_in(filename);
  auto points = ply_in.getVertexPositions();
  auto colors = ply_in.getVertexColors();
  log_info("Loaded PLY file {} with {} points and {} colors", filename, points.size(), colors.size());
  PointCloud pc;
  pc.points.reserve(points.size());
  pc.colors.reserve(colors.size());

  for (size_t i = 0; i < points.size(); i++) {
    pc.points.push_back({(float)points[i][0], (float)points[i][1], (float)points[i][2]});
    pc.colors.push_back({
      (float) colors[i][0] / RGB_NORMALIZATION_FACTOR,
      (float) colors[i][1] / RGB_NORMALIZATION_FACTOR,
      (float) colors[i][2] / RGB_NORMALIZATION_FACTOR});
  }
  return pc;
}

PointCloud load_point_cloud(const std::string& filename) {
  if (filename.size() >= 4) {
    std::string ext = filename.substr(filename.size() - 4);
    if (ext == ".ply" || ext == ".PLY") {
      return load_ply(filename);
    } else if (ext == ".txt" || ext == ".TXT") {
      return load_from_colmap(filename);
    } else {
      log_error("Unsupported point cloud file format: {}", ext);
      return PointCloud();
    }
  } else {
    log_error("Filename too short to determine format: {}", filename);
    return PointCloud();
  }
}

void save_ply(const std::string& filename, const Gaussian3d& gs, bool full_features) {
  auto& sh_dc = gs.sh0;
  auto& xyz = gs.means;
  auto& scal = gs.scales;
  auto& opa = gs.opacities;
  auto& rot = gs.rotations;

  // Create PLY data object
  happly::PLYData plyData;

  // Add vertex element
  size_t num_points = xyz.size();
  plyData.addElement("vertex", num_points);

  {  // Extract and add xyz coordinates
    std::vector<float> x, y, z;
    x.reserve(num_points);
    y.reserve(num_points);
    z.reserve(num_points);

    for (const auto& pos : xyz) {
      x.push_back(pos.x);
      y.push_back(pos.y);
      z.push_back(pos.z);
    }

    plyData.getElement("vertex").addProperty<float>("x", x);
    plyData.getElement("vertex").addProperty<float>("y", y);
    plyData.getElement("vertex").addProperty<float>("z", z);
  }
  { // Extract and add SH coefficients (DC components)
    std::vector<float> f_dc_0, f_dc_1, f_dc_2;
    f_dc_0.reserve(num_points);
    f_dc_1.reserve(num_points);
    f_dc_2.reserve(num_points);

    for (const auto& dc : sh_dc) {
      f_dc_0.push_back(dc.x);
      f_dc_1.push_back(dc.y);
      f_dc_2.push_back(dc.z);
    }
    
    std::vector<unsigned char> f_dc_r(num_points);
    std::vector<unsigned char> f_dc_g(num_points);
    std::vector<unsigned char> f_dc_b(num_points);
    for (size_t i = 0; i < num_points; i++) {
      constexpr float k_inv_sh = 0.28209479177387814f;
      f_dc_r[i] = (unsigned char)(clamp(f_dc_0[i] * k_inv_sh + 0.5f, 0.0f, 1.0f) * RGB_NORMALIZATION_FACTOR);
      f_dc_g[i] = (unsigned char)(clamp(f_dc_1[i] * k_inv_sh + 0.5f, 0.0f, 1.0f) * RGB_NORMALIZATION_FACTOR);
      f_dc_b[i] = (unsigned char)(clamp(f_dc_2[i] * k_inv_sh + 0.5f, 0.0f, 1.0f) * RGB_NORMALIZATION_FACTOR);
    }
    plyData.getElement("vertex").addProperty<unsigned char>("red", f_dc_r);
    plyData.getElement("vertex").addProperty<unsigned char>("green", f_dc_g);
    plyData.getElement("vertex").addProperty<unsigned char>("blue", f_dc_b);
    if (! full_features) {
      log_info("Saved {} gaussians to PLY file: {} (partial features)", num_points, filename);
      plyData.write(filename, happly::DataFormat::Binary);
      return;
    }
    plyData.getElement("vertex").addProperty<float>("f_dc_r", f_dc_0);
    plyData.getElement("vertex").addProperty<float>("f_dc_g", f_dc_1);
    plyData.getElement("vertex").addProperty<float>("f_dc_b", f_dc_2);
  }
  // Extract and add rest of SH coefficients from per-degree buffers (sh1, sh2, sh3)
  // The PLY format stores them as a flat sequence of 15 coefficients per Gaussian.
  // Degree 1: 3 coeffs (indices 0-2), Degree 2: 5 coeffs (indices 3-7), Degree 3: 7 coeffs (indices 8-14).
  {
    // Map from flat rest index [0..14] to (degree, coeff_within_degree)
    struct ShRestMapping { int degree; int coeff; };
    constexpr ShRestMapping rest_map[15] = {
      {1,0},{1,1},{1,2},
      {2,0},{2,1},{2,2},{2,3},{2,4},
      {3,0},{3,1},{3,2},{3,3},{3,4},{3,5},{3,6}
    };
    const std::vector<vec3>* sh_degree_bufs[4] = {&gs.sh0, &gs.sh1, &gs.sh2, &gs.sh3};

    constexpr int num_rest_coeffs = kMaxSphericalHarmonicsCoefficients - 1; // 15
    for (int coeff_idx = 0; coeff_idx < num_rest_coeffs; ++coeff_idx) {
      std::vector<float> f_rest_x, f_rest_y, f_rest_z;
      f_rest_x.reserve(num_points);
      f_rest_y.reserve(num_points);
      f_rest_z.reserve(num_points);

      const auto& mapping = rest_map[coeff_idx];
      const auto& degree_buf = *sh_degree_bufs[mapping.degree];
      const int num_coeffs_in_degree = kSHDegreeNumCoeffs[mapping.degree];

      for (size_t point_idx = 0; point_idx < num_points; ++point_idx) {
        size_t array_idx = point_idx * num_coeffs_in_degree + mapping.coeff;
        if (array_idx < degree_buf.size()) {
          const vec3& coeff = degree_buf[array_idx];
          f_rest_x.push_back(coeff.x);
          f_rest_y.push_back(coeff.y);
          f_rest_z.push_back(coeff.z);
        } else {
          f_rest_x.push_back(0.0f);
          f_rest_y.push_back(0.0f);
          f_rest_z.push_back(0.0f);
        }
      }

      plyData.getElement("vertex").addProperty<float>("f_rest_r_" + std::to_string(coeff_idx), f_rest_x);
      plyData.getElement("vertex").addProperty<float>("f_rest_g_" + std::to_string(coeff_idx), f_rest_y);
      plyData.getElement("vertex").addProperty<float>("f_rest_b_" + std::to_string(coeff_idx), f_rest_z);
    }
  }

  // Add opacity
  plyData.getElement("vertex").addProperty<float>("opacity", opa);

  // Extract and add scale
  std::vector<float> scale_0, scale_1, scale_2;
  scale_0.reserve(num_points);
  scale_1.reserve(num_points);
  scale_2.reserve(num_points);

  for (const auto& s : scal) {
    scale_0.push_back(s.x);
    scale_1.push_back(s.y);
    scale_2.push_back(s.z);
  }

  plyData.getElement("vertex").addProperty<float>("scale_0", scale_0);
  plyData.getElement("vertex").addProperty<float>("scale_1", scale_1);
  plyData.getElement("vertex").addProperty<float>("scale_2", scale_2);

  // Extract and add rotation
  std::vector<float> rot_0, rot_1, rot_2, rot_3;
  rot_0.reserve(num_points);
  rot_1.reserve(num_points);
  rot_2.reserve(num_points);
  rot_3.reserve(num_points);

  for (const auto& r : rot) {
    rot_0.push_back(r.x);
    rot_1.push_back(r.y);
    rot_2.push_back(r.z);
    rot_3.push_back(r.w);
  }

  plyData.getElement("vertex").addProperty<float>("rot_w", rot_0);
  plyData.getElement("vertex").addProperty<float>("rot_x", rot_1);
  plyData.getElement("vertex").addProperty<float>("rot_y", rot_2);
  plyData.getElement("vertex").addProperty<float>("rot_z", rot_3);

  // Write to file
  plyData.write(filename, happly::DataFormat::Binary);
  
  log_info("Saved {} gaussians to PLY file: {}", num_points, filename);
}

}  // namespace tinygs
