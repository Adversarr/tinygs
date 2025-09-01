#pragma once

#include <vector>

#include "tinygs/common.hpp"
#include "tinygs/initialization/initialization.hpp"

namespace tinygs {
struct KnnParameters {
  int num_neighbors = 3;           // Number of neighbors to consider for distance calculation
  float min_distance = 1e-7f;      // Minimum distance threshold
  float default_distance = 0.01f;  // Default distance for edge cases
  float init_scaling = 0.6f;       // Initial scaling factor
  float init_opacity = 0.5f;       // Initial opacity value
  int sh_degree = 3;               // Spherical harmonics degree
  bool use_random_init = false;    // Whether to use random initialization
  int random_num_points = 100000;  // Number of points for random initialization
  float random_extent = 6.0f;      // Extent for random initialization
};

// Initialize the GS size to be the average dist of the K nearest neighbors
class KnnInitialization : public InitializationBase {
private:
  KnnParameters m_params;

  // Core KNN functionality
  std::vector<float> compute_mean_neighbor_distances(const std::vector<vec3>& points) const;
  float calculate_scene_scale(const std::vector<vec3>& points, const vec3& center) const;
  vec3 rgb_to_sh(const vec3& rgb) const;

public:
  explicit KnnInitialization(const KnnParameters& params = KnnParameters{});

  void initialize(PointCloud& pointcloud) override;

  // Parameter accessors
  void set_parameters(const KnnParameters& params) { m_params = params; }
  const KnnParameters& get_parameters() const { return m_params; }
};
}  // namespace tinygs
