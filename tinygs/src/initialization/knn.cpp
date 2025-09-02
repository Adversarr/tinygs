/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "cuda/common_host.hpp"
#include "nanoflann.hpp"
#include "tinygs/initialization/knn.hpp"
#include "utils/scope_timer.hpp"
#include <nlohmann/json.hpp>

namespace tinygs {

// Point cloud adaptor for nanoflann
struct PointCloudAdaptor {
  const std::vector<vec3>& points;

  explicit PointCloudAdaptor(const std::vector<vec3>& pts) : points(pts) {}

  inline size_t kdtree_get_point_count() const { return points.size(); }

  inline float kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }
};

using KDTree
    = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3>;

KnnInitialization::KnnInitialization(const KnnParameters& params) : m_params(params) {
}

std::vector<float> KnnInitialization::compute_mean_neighbor_distances(const std::vector<vec3>& points) const {
  const size_t num_points = points.size();
  std::vector<float> result(num_points, m_params.default_distance);

  if (num_points <= 1) {
    return result;
  }

  PointCloudAdaptor cloud(points);
  KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

#pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    const float query_pt[3] = {points[i].x, points[i].y, points[i].z};

    const size_t num_results = std::min(static_cast<size_t>(m_params.num_neighbors + 1), num_points);
    std::vector<size_t> ret_indices(num_results);
    std::vector<float> out_dists_sqr(num_results);

    nanoflann::KNNResultSet<float> result_set(num_results);
    result_set.init(&ret_indices[0], &out_dists_sqr[0]);
    index.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParameters(10));

    float sum_dist = 0.0f;
    int valid_neighbors = 0;

    // Skip the first result (self) and collect neighbors
    for (size_t j = 1; j < num_results && valid_neighbors < m_params.num_neighbors; ++j) {
      if (out_dists_sqr[j] > m_params.min_distance * m_params.min_distance) {
        sum_dist += std::sqrt(out_dists_sqr[j]);
        valid_neighbors++;
      }
    }

    result[i] = (valid_neighbors > 0) ? (sum_dist / valid_neighbors) : m_params.default_distance;
  }

  return result;
}

float KnnInitialization::calculate_scene_scale(const std::vector<vec3>& points, const vec3& center) const {
  if (points.empty()) {
    return 1.0f;
  }

  std::vector<float> distances;
  distances.reserve(points.size());

  for (const auto& point : points) {
    distances.push_back(length(point - center));
  }

  std::sort(distances.begin(), distances.end());

  // Return median distance
  size_t mid = distances.size() / 2;
  if (distances.size() % 2 == 0) {
    return (distances[mid - 1] + distances[mid]) / 2.0f;
  } else {
    return distances[mid];
  }
}

vec3 KnnInitialization::rgb_to_sh(const vec3& rgb) const {
  constexpr float kInvSH = 0.28209479177387814f;
  return (rgb - vec3(0.5f)) / kInvSH;
}

void KnnInitialization::initialize(const PointCloud& pointcloud) {
  TINYGS_TIMER("KnnInitialization::initialize");
  std::vector<vec3> positions;
  std::vector<vec3> colors;

  if (m_params.use_random_init) {
    // Generate random points
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    positions.reserve(m_params.random_num_points);
    colors.reserve(m_params.random_num_points);

    for (int i = 0; i < m_params.random_num_points; ++i) {
      positions.emplace_back(dis(gen) * m_params.random_extent, dis(gen) * m_params.random_extent,
                             dis(gen) * m_params.random_extent);

      std::uniform_real_distribution<float> color_dis(0.0f, 1.0f);
      colors.emplace_back(color_dis(gen), color_dis(gen), color_dis(gen));
    }
  } else {
    // Use existing point cloud data
    positions = pointcloud.points;
    colors.reserve(pointcloud.colors.size());

    // Normalize colors from [0, 255] to [0, 1]
    for (const auto& color : pointcloud.colors) {
      colors.emplace_back(color / 255.0f);
    }
  }

  if (positions.empty()) {
    std::cerr << "Error: No points to initialize from" << std::endl;
    return;
  }

  // Calculate scene center and scale
  vec3 scene_center(0.0f);
  for (const auto& pos : positions) {
    scene_center += pos;
  }
  scene_center /= static_cast<float>(positions.size());

  float scene_scale = calculate_scene_scale(positions, scene_center);

  // Scale positions if using random initialization
  if (m_params.use_random_init) {
    for (auto& pos : positions) {
      pos *= scene_scale;
    }
  }

  // Compute neighbor distances for scaling initialization
  auto neighbor_distances = compute_mean_neighbor_distances(positions);

  // Clear existing gaussians and resize to fit new data
  const size_t num_points = positions.size();
  m_gaussians.means_opacities.resize(num_points);
  m_gaussians.rotations.resize(num_points);
  m_gaussians.scales.resize(num_points);
  m_gaussians.sh_coefficients.resize(num_points * kMaxSphericalHarmonicsCoefficients);

  // Initialize gaussians using SoA structure
#pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    // Set position and opacity
    m_gaussians.means_opacities[i] = vec4(positions[i].x, positions[i].y, positions[i].z, m_params.init_opacity);

    // Set rotation (identity quaternion: w=1, x=0, y=0, z=0)
    m_gaussians.rotations[i] = vec4(1.0f, 0.0f, 0.0f, 0.0f);

    // Set scale based on neighbor distances
    float scale_value = std::max(neighbor_distances[i] * m_params.init_scaling, m_params.min_distance);
    float log_scale = std::log(scale_value);
    m_gaussians.scales[i] = vec3(log_scale, log_scale, log_scale);

    // Set spherical harmonics coefficients
    // vec3 sh_color = rgb_to_sh(colors[i]);
    vec3 sh_color = colors[i]; // Use raw color for better initialization
    m_gaussians.sh_coefficients[i * kMaxSphericalHarmonicsCoefficients] = sh_color;
    for (int j = 1; j < kMaxSphericalHarmonicsCoefficients; ++j) {
      m_gaussians.sh_coefficients[i * kMaxSphericalHarmonicsCoefficients + j] = vec3(0.0f);
    }

    // Initialize SH coefficients array
    for (int j = 1; j < kMaxSphericalHarmonicsCoefficients; ++j) {
      m_gaussians.sh_coefficients[i * kMaxSphericalHarmonicsCoefficients + j] = vec3(0.0f);
    }
  }

  log_info("Initialized {} gaussians with KNN method", m_gaussians.means_opacities.size());
  log_info("Scene scale: {}", scene_scale);
  log_info("SH degree: {}", m_params.sh_degree);
}

void KnnInitialization::set_parameters(const json& params) {
  if (params.contains("num_neighbors")) {
    m_params.num_neighbors = params["num_neighbors"].get<int>();
  }
  if (params.contains("min_distance")) {
    m_params.min_distance = params["min_distance"].get<float>();
  }
  if (params.contains("default_distance")) {
    m_params.default_distance = params["default_distance"].get<float>();
  }
  if (params.contains("init_scaling")) {
    m_params.init_scaling = params["init_scaling"].get<float>();
  }
  if (params.contains("init_opacity")) {
    m_params.init_opacity = params["init_opacity"].get<float>();
  }
  if (params.contains("sh_degree")) {
    m_params.sh_degree = params["sh_degree"].get<int>();
  }
  if (params.contains("use_random_init")) {
    m_params.use_random_init = params["use_random_init"].get<bool>();
  }
  if (params.contains("random_num_points")) {
    m_params.random_num_points = params["random_num_points"].get<int>();
  }
  if (params.contains("random_extent")) {
    m_params.random_extent = params["random_extent"].get<float>();
  }
}

json KnnInitialization::get_parameters() const {
  json params;
  params["num_neighbors"] = m_params.num_neighbors;
  params["min_distance"] = m_params.min_distance;
  params["default_distance"] = m_params.default_distance;
  params["init_scaling"] = m_params.init_scaling;
  params["init_opacity"] = m_params.init_opacity;
  params["sh_degree"] = m_params.sh_degree;
  params["use_random_init"] = m_params.use_random_init;
  params["random_num_points"] = m_params.random_num_points;
  params["random_extent"] = m_params.random_extent;
  return params;
}

}  // namespace tinygs