#pragma once

#include <vector>

#include "tinygs/common.hpp"
#include "tinygs/initialization/initialization.hpp"

namespace tinygs {

struct RandomParameters {
  int num_points = 100000;         // Number of random points to generate
  float extent = 6.0f;             // Spatial extent for random point generation
  float init_scaling = 0.01f;      // Initial scaling factor for gaussians
  float init_opacity = 0.5f;       // Initial opacity value
  int sh_degree = 3;               // Spherical harmonics degree
  float min_scale = 1e-7f;         // Minimum scale value
  float max_scale = 1.0f;          // Maximum scale value
  bool use_uniform_scale = true;   // Whether to use uniform scaling or random scaling
  unsigned int seed = 42;          // Random seed for reproducibility
};

// Initialize gaussians with random positions, colors, and properties
class RandomInitialization : public InitializationBase {
private:
  RandomParameters m_params;

  // Helper functions
  vec3 rgb_to_sh(const vec3& rgb) const;
  vec3 generate_random_color() const;
  vec3 generate_random_position() const;
  vec3 generate_random_scale() const;

public:
  explicit RandomInitialization(const RandomParameters& params = RandomParameters{});

  void initialize(const PointCloud& /*pointcloud*/) override;

  // Parameter accessors
  void set_random_parameters(const RandomParameters& params) { m_params = params; }
  const RandomParameters& get_random_parameters() const { return m_params; }

  // JSON interface overrides
  void set_params(const json& params) override;
  json get_params() const override;
};

}  // namespace tinygs