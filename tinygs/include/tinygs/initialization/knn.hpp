#pragma once

#include <vector>

#include "tinygs/common.hpp"
#include "tinygs/initialization/initialization.hpp"

namespace tinygs {

struct KnnParameters {
  int num_neighbors = 4;           // Number of neighbors to consider for distance calculation
  float min_distance = 1e-7f;      // Minimum distance threshold
  float max_distance = 1e-2f;      // Maximum distance threshold
  float default_distance = 0.001f; // Default distance for edge cases
  float init_scaling = 1.0f;       // Initial scaling factor
  float init_opacity = 0.1f;       // Initial opacity value
  int sh_degree = 3;               // Spherical harmonics degree
  
  // Radius outlier removal parameters
  bool enable_radius_outlier_removal = true; // Enable/disable radius outlier removal
  int nb_points = 16;              // Minimum number of neighbors within radius
  float radius = 0.05f;            // Radius for neighbor search
};

// Initialize the GS size to be the average dist of the K nearest neighbors
class KnnInitialization : public InitializationBase {
private:
  KnnParameters m_params;

  // Core KNN functionality
  std::vector<float> compute_mean_neighbor_distances(const std::vector<vec3>& points) const;
  float calculate_scene_scale(const std::vector<float>& distances) const;
  vec3 rgb_to_sh(const vec3& rgb) const;
  
  // Radius outlier removal functionality
  std::vector<size_t> radius_outlier_removal(const std::vector<vec3>& points) const;

public:
  explicit KnnInitialization(const KnnParameters& params = KnnParameters{});

  void initialize(const PointCloud& pointcloud) override;

  // Parameter accessors
  void set_knn_parameters(const KnnParameters& params) { m_params = params; }
  const KnnParameters& get_knn_parameters() const { return m_params; }

  float get_scene_scale() const { return m_scene_scale; }

  // JSON interface overrides
  void set_params(const json& params) override;
  json get_params() const override;

protected:
  float m_scene_scale = 1.0f;
};
}  // namespace tinygs
