#pragma once

#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct AdanParameters {
  // Adan coefficients
  float beta1 = 0.98f;
  float beta2 = 0.92f;
  float beta3 = 0.99f;
  float epsilon = 1e-8f;

  AdanParameters() = default;
  explicit AdanParameters(const json &config) { from_json(config); }

  json to_json() const {
    json j;
    j["beta1"] = beta1;
    j["beta2"] = beta2;
    j["beta3"] = beta3;
    j["epsilon"] = epsilon;
    return j;
  }

  void from_json(const json &config) {
    if (config.contains("beta1")) beta1 = config.at("beta1").get<float>();
    if (config.contains("beta2")) beta2 = config.at("beta2").get<float>();
    if (config.contains("beta3")) beta3 = config.at("beta3").get<float>();
    if (config.contains("epsilon")) epsilon = config.at("epsilon").get<float>();
  }
};

// Adan optimizer for Gaussian Splatting
class Adan final : public OptimizerBase {
public:
  Adan(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);
  ~Adan() override = default;

  void reset() override;
  void step(float scale, cudaStream_t stream) override;
  void remove(char* kept_flag, int num_kept) override;
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  void reset(int* indices, int num_reset) override;
  void reset_opacity() override;
  void reorder(uint* indices) override;

  void set_params(const json& config) override;
  json get_params() const override;

private:
  AdanParameters m_adan_params;
  uint32_t m_global_steps = 0;

  // First/second/diff moments and previous gradients for each parameter type
  thrust::device_vector<vec3> m_means_m;      // exp_avg
  thrust::device_vector<vec3> m_means_v;      // exp_avg_sq
  thrust::device_vector<vec3> m_means_d;      // exp_avg_diff
  thrust::device_vector<vec3> m_means_prev_g; // previous grad

  thrust::device_vector<float> m_opacities_m;
  thrust::device_vector<float> m_opacities_v;
  thrust::device_vector<float> m_opacities_d;
  thrust::device_vector<float> m_opacities_prev_g;

  thrust::device_vector<vec4> m_rotations_m;
  thrust::device_vector<vec4> m_rotations_v;
  thrust::device_vector<vec4> m_rotations_d;
  thrust::device_vector<vec4> m_rotations_prev_g;

  thrust::device_vector<vec3> m_scales_m;
  thrust::device_vector<vec3> m_scales_v;
  thrust::device_vector<vec3> m_scales_d;
  thrust::device_vector<vec3> m_scales_prev_g;

  thrust::device_vector<vec3> m_sh0_m;
  thrust::device_vector<vec3> m_sh0_v;
  thrust::device_vector<vec3> m_sh0_d;
  thrust::device_vector<vec3> m_sh0_prev_g;

  thrust::device_vector<vec3> m_shrest_m;
  thrust::device_vector<vec3> m_shrest_v;
  thrust::device_vector<vec3> m_shrest_d;
  thrust::device_vector<vec3> m_shrest_prev_g;
};

} // namespace tinygs
