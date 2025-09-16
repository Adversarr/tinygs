/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tinygs/initialization/knn.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>

#include "cuda/common_host.hpp"
#include "nanoflann.hpp"
#include "random/pcg32.hpp"
#include "utils/scope_timer.hpp"

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

// #pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    const float query_pt[3] = {points[i].x, points[i].y, points[i].z};

    const size_t num_results = std::min(static_cast<size_t>(m_params.num_neighbors + 1), num_points);
    std::vector<size_t> ret_indices(num_results);
    std::vector<float> out_dists_sqr(num_results, 0);

    nanoflann::KNNResultSet<float> result_set(num_results);
    result_set.init(&ret_indices[0], &out_dists_sqr[0]);
    index.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParameters(10));

    float sum_dist = 0.0f;
    int valid_neighbors = 0;

    // Skip the first result (self) and collect neighbors
    for (size_t j = 1; j < num_results && valid_neighbors < m_params.num_neighbors; ++j) {
      if (out_dists_sqr[j] > 1e-8f) {
        sum_dist += out_dists_sqr[j];
        valid_neighbors++;
      }
    }

    result[i] = (valid_neighbors > 0) ? sqrtf(sum_dist / valid_neighbors) : m_params.default_distance;
    result[i] = std::clamp(result[i], m_params.min_distance, m_params.max_distance);

    if (valid_neighbors == 0) {
      log_warning("No valid neighbors for point {}: {:.3e} {:.3e} {:.3e}", i,
        query_pt[0], query_pt[1], query_pt[2]);
    }
  }

  return result;
}

std::vector<size_t> KnnInitialization::radius_outlier_removal(const std::vector<vec3>& points) const {
  std::vector<size_t> valid_indices;
  
  if (!m_params.enable_radius_outlier_removal || points.empty()) {
    // Return all indices if outlier removal is disabled
    valid_indices.resize(points.size());
    std::iota(valid_indices.begin(), valid_indices.end(), 0);
    return valid_indices;
  }

  const size_t num_points = points.size();
  
  // Build KD-tree for efficient radius search
  PointCloudAdaptor cloud(points);
  KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

  // Check each point for sufficient neighbors within radius
  for (size_t i = 0; i < num_points; ++i) {
    const float query_pt[3] = {points[i].x, points[i].y, points[i].z};
    
    // Use KNN search with a large number to find all potential neighbors
    const size_t max_neighbors = std::min(num_points, static_cast<size_t>(1000));
    std::vector<size_t> ret_indices(max_neighbors);
    std::vector<float> out_dists_sqr(max_neighbors);
    
    nanoflann::KNNResultSet<float> result_set(max_neighbors);
    result_set.init(&ret_indices[0], &out_dists_sqr[0]);
    index.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParameters(10));
    
    // Count neighbors within radius (excluding self)
    int neighbor_count = 0;
    const float radius_sqr = m_params.radius * m_params.radius;
    for (size_t j = 0; j < max_neighbors; ++j) {
      if (ret_indices[j] != i && out_dists_sqr[j] <= radius_sqr) {
        neighbor_count++;
      }
    }
    
    // Keep point if it has enough neighbors
    if (neighbor_count >= m_params.nb_points) {
      valid_indices.push_back(i);
    }
  }
  
  log_info("Radius outlier removal: kept {} out of {} points", valid_indices.size(), num_points);
  return valid_indices;
}

vec3 KnnInitialization::rgb_to_sh(const vec3& rgb) const {
  constexpr float kInvSH = 0.28209479177387814f;
  return (rgb - vec3(0.5f)) / kInvSH;
}

void KnnInitialization::initialize(const PointCloud& pointcloud) {
  TINYGS_TIMER("KnnInitialization::initialize");
  const auto& positions = pointcloud.points;
  const auto& colors = pointcloud.colors;

  if (positions.empty()) {
    std::cerr << "Error: No points to initialize from" << std::endl;
    return;
  }

  // Apply radius outlier removal if enabled
  auto valid_indices = radius_outlier_removal(positions);
  
  // Create filtered point cloud
  std::vector<vec3> filtered_positions;
  std::vector<vec3> filtered_colors;
  filtered_positions.reserve(valid_indices.size());
  filtered_colors.reserve(valid_indices.size());
  
  for (size_t idx : valid_indices) {
    filtered_positions.push_back(positions[idx]);
    filtered_colors.push_back(colors[idx]);
  }

  // Calculate scene center and scale using filtered points
  vec3 scene_center(0.0f);
  for (const auto& pos : filtered_positions) {
    scene_center += pos;
  }
  scene_center /= static_cast<float>(filtered_positions.size());

  // Compute neighbor distances for scaling initialization using filtered points
  auto neighbor_distances = compute_mean_neighbor_distances(filtered_positions);

  // Clear existing gaussians and resize to fit filtered data
  const size_t num_points = filtered_positions.size();
  m_gaussians.opacities.resize(num_points);
  m_gaussians.means.resize(num_points);
  m_gaussians.rotations.resize(num_points, vec4(1.0f, 0.0f, 0.0f, 0.0f));
  m_gaussians.scales.resize(num_points);
  m_gaussians.sh_coefficient_0.resize(num_points);
  m_gaussians.sh_coefficients_rest.resize(num_points * (kMaxSphericalHarmonicsCoefficients - 1));

  auto init_opa = log(m_params.init_opacity / (1 - m_params.init_opacity));
  // Initialize gaussians using SoA structure with filtered points
  for (size_t i = 0; i < num_points; ++i) {
    // Set position and opacity
    m_gaussians.means[i] = vec3(filtered_positions[i].x, filtered_positions[i].y, filtered_positions[i].z);
    m_gaussians.opacities[i] = init_opa;

    // Set rotation (identity quaternion: w=1, x=0, y=0, z=0)
    m_gaussians.rotations[i] = vec4(1.0f, 0.0f, 0.0f, 0.0f);

    // Set scale based on neighbor distances
    float scale_value = std::max(neighbor_distances[i] * m_params.init_scaling, m_params.min_distance);
    float log_scale = deactivate_scale(scale_value);
    m_gaussians.scales[i] = vec3(log_scale, log_scale, log_scale);

    // Set spherical harmonics coefficients
    vec3 sh_color = rgb_to_sh(filtered_colors[i]);
    m_gaussians.sh_coefficient_0[i] = sh_color;

    // Initialize SH coefficients rest array
    for (int j = 0; j < kMaxSphericalHarmonicsCoefficients - 1; ++j) {
      m_gaussians.sh_coefficients_rest[i * (kMaxSphericalHarmonicsCoefficients - 1) + j] = vec3(0.0f);
    }
  }

  log_info("Initialized {} gaussians with KNN method", m_gaussians.means.size());
  log_info("SH degree: {}", m_params.sh_degree);
}

void KnnInitialization::set_params(const json& params) {
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
  if (params.contains("enable_radius_outlier_removal")) {
    m_params.enable_radius_outlier_removal = params["enable_radius_outlier_removal"].get<bool>();
  }
  if (params.contains("nb_points")) {
    m_params.nb_points = params["nb_points"].get<int>();
  }
  if (params.contains("radius")) {
    m_params.radius = params["radius"].get<float>();
  }
}

json KnnInitialization::get_params() const {
  json params;
  params["type"] = "knn";
  params["num_neighbors"] = m_params.num_neighbors;
  params["min_distance"] = m_params.min_distance;
  params["default_distance"] = m_params.default_distance;
  params["init_scaling"] = m_params.init_scaling;
  params["init_opacity"] = m_params.init_opacity;
  params["sh_degree"] = m_params.sh_degree;
  params["enable_radius_outlier_removal"] = m_params.enable_radius_outlier_removal;
  params["nb_points"] = m_params.nb_points;
  params["radius"] = m_params.radius;
  return params;
}

}  // namespace tinygs