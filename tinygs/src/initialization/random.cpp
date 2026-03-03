/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cmath>
#include <random>

#include "cuda/common_host.hpp"
#include "tinygs/initialization/random.hpp"
#include "utils/scope_timer.hpp"
#include <nlohmann/json.hpp>

namespace tinygs {

RandomInitialization::RandomInitialization(const RandomParameters& params) : m_params(params) {
}

vec3 RandomInitialization::rgb_to_sh(const vec3& rgb) const {
  constexpr float kInvSH = 0.28209479177387814f;
  return (rgb - vec3(0.5f)) / kInvSH;
}

void RandomInitialization::initialize(const PointCloud&  /*pointcloud*/) {
  ScopeTimer timer("RandomInitialization::initialize");
  
  // Generate random positions and colors
  std::vector<vec3> positions;
  std::vector<vec3> colors;
  
  positions.reserve(m_params.num_points);
  colors.reserve(m_params.num_points);
  
  // Use separate generators for reproducibility
  std::mt19937 pos_gen(m_params.seed);
  std::mt19937 color_gen(m_params.seed + 1000);
  
  std::uniform_real_distribution<float> pos_dis(-m_params.extent, m_params.extent);
  std::uniform_real_distribution<float> color_dis(0.0f, 1.0f);
  
  for (int i = 0; i < m_params.num_points; ++i) {
    // Generate random position
    positions.emplace_back(pos_dis(pos_gen), pos_dis(pos_gen), pos_dis(pos_gen));
    
    // Generate random color
    colors.emplace_back(color_dis(color_gen), color_dis(color_gen), color_dis(color_gen));
  }
  
  // Clear existing gaussians and resize to fit new data
  const size_t num_points = positions.size();
  m_gaussians.means.resize(num_points);
  m_gaussians.opacities.resize(num_points);
  m_gaussians.rotations.resize(num_points);
  m_gaussians.scales.resize(num_points);
  m_gaussians.sh0.resize(num_points);
  m_gaussians.sh1.resize(num_points * kSHDegreeNumCoeffs[1], vec3(0.0f));
  m_gaussians.sh2.resize(num_points * kSHDegreeNumCoeffs[2], vec3(0.0f));
  m_gaussians.sh3.resize(num_points * kSHDegreeNumCoeffs[3], vec3(0.0f));
  
  // Initialize gaussians using SoA structure
#pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    // Set position and opacity
    m_gaussians.means[i] = vec3(positions[i].x, positions[i].y, positions[i].z);
    m_gaussians.opacities[i] = m_params.init_opacity;
    
    // Set rotation (identity quaternion: w=1, x=0, y=0, z=0)
    m_gaussians.rotations[i] = vec4(1.0f, 0.0f, 0.0f, 0.0f);
    
    // Set scale
    vec3 scale;
    if (m_params.use_uniform_scale) {
      float log_scale = std::log(m_params.init_scaling);
      scale = vec3(log_scale, log_scale, log_scale);
    } else {
      // Use thread-local generator for parallel safety
      thread_local std::mt19937 scale_gen(m_params.seed + 2000 + i);
      std::uniform_real_distribution<float> scale_dis(m_params.min_scale, m_params.max_scale);
      scale = vec3(std::log(scale_dis(scale_gen)), 
                   std::log(scale_dis(scale_gen)), 
                   std::log(scale_dis(scale_gen)));
    }
    m_gaussians.scales[i] = scale;
    
    // Set spherical harmonics coefficients (degree 0 = DC term)
    vec3 sh_color = rgb_to_sh(colors[i]);
    m_gaussians.sh0[i] = sh_color;
    
    // Higher-degree SH already initialized to zero during resize
  }
  
  log_info("Initialized {} gaussians with random method", m_gaussians.means.size());
  log_info("Extent: {}", m_params.extent);
  log_info("SH degree: {}", m_params.sh_degree);
  log_info("Uniform scale: {}", m_params.use_uniform_scale);
}

void RandomInitialization::set_params(const json& params) {
  if (params.contains("num_points")) {
    m_params.num_points = params["num_points"].get<int>();
  }
  if (params.contains("extent")) {
    m_params.extent = params["extent"].get<float>();
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
  if (params.contains("min_scale")) {
    m_params.min_scale = params["min_scale"].get<float>();
  }
  if (params.contains("max_scale")) {
    m_params.max_scale = params["max_scale"].get<float>();
  }
  if (params.contains("use_uniform_scale")) {
    m_params.use_uniform_scale = params["use_uniform_scale"].get<bool>();
  }
  if (params.contains("seed")) {
    m_params.seed = params["seed"].get<unsigned int>();
  }
}

json RandomInitialization::get_params() const {
  json params;
  params["type"] = "random";
  params["num_points"] = m_params.num_points;
  params["extent"] = m_params.extent;
  params["init_scaling"] = m_params.init_scaling;
  params["init_opacity"] = m_params.init_opacity;
  params["sh_degree"] = m_params.sh_degree;
  params["min_scale"] = m_params.min_scale;
  params["max_scale"] = m_params.max_scale;
  params["use_uniform_scale"] = m_params.use_uniform_scale;
  params["seed"] = m_params.seed;
  return params;
}

}  // namespace tinygs