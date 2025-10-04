#pragma once

#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"
#include "tinygs/common.hpp"

namespace tinygs {

struct LambParameters {
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;
  float trust_ratio_min = 0.01f;
  float trust_ratio_max = 10.0f;

  LambParameters() = default;
  explicit LambParameters(const json& config) { from_json(config); }

  json to_json() const;

  void from_json(const json &config);
};

/// Layer-wise Adaptive Moments (LAMB) optimizer
class Lamb final : public OptimizerBase {
public:
  Lamb(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);
  ~Lamb() override = default;

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
  LambParameters m_lamb_params;
  uint32_t m_global_steps = 0;

  // First/second moment estimates for each parameter type
  thrust::device_vector<vec3> m_means_first;
  thrust::device_vector<vec3> m_means_second;
  thrust::device_vector<float> m_opacities_first;
  thrust::device_vector<float> m_opacities_second;
  thrust::device_vector<vec4> m_rotations_first;
  thrust::device_vector<vec4> m_rotations_second;
  thrust::device_vector<vec3> m_scales_first;
  thrust::device_vector<vec3> m_scales_second;
  thrust::device_vector<vec3> m_sh_coefficient_0_first;
  thrust::device_vector<vec3> m_sh_coefficient_0_second;
  thrust::device_vector<vec3> m_sh_coefficients_rest_first;
  thrust::device_vector<vec3> m_sh_coefficients_rest_second;
};

} // namespace tinygs