#pragma once

#include "tinygs/common.hpp"
#include "tinygs/optim/adam.hpp"
#include "tinygs/optim/optim.hpp"

namespace tinygs {

/// @brief Adam/AdamW variant with shared per-Gaussian iteration counters.
///
/// Bias correction uses a per-Gaussian step counter t. New Gaussians created by
/// duplication start with t=0 and are incremented on the first optimizer step.
class AdamPerGaussian final : public OptimizerBase {
public:
  AdamPerGaussian(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad);

  ~AdamPerGaussian() override = default;

  void reset() override;

  void step(float scale, BackendStream stream) override;
  void step(const GroupStepConfig& step_config, BackendStream stream) override;

  void remove(char* kept_flag, int num_kept) override;
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  void reset(int* indices, int num_reset) override;
  void reset_opacity() override;
  void reorder(uint* indices) override;

  void set_params(const json& config) override;
  json get_params() const override;

private:
  void step_adam(float scale, BackendStream stream);
  void step_adamw(float scale, BackendStream stream);

  AdamParameters m_adam_params;
  thrust::device_vector<uint32_t> m_steps;

  thrust::device_vector<vec3> m_means_first;
  thrust::device_vector<float> m_opacities_first;
  thrust::device_vector<vec4> m_rotations_first;
  thrust::device_vector<vec3> m_scales_first;
  thrust::device_vector<float> m_sh0_first;
  thrust::device_vector<float> m_sh1_first;
  thrust::device_vector<float> m_sh2_first;
  thrust::device_vector<float> m_sh3_first;

  thrust::device_vector<vec3> m_means_second;
  thrust::device_vector<float> m_opacities_second;
  thrust::device_vector<vec4> m_rotations_second;
  thrust::device_vector<vec3> m_scales_second;
  thrust::device_vector<float> m_sh0_second;
  thrust::device_vector<float> m_sh1_second;
  thrust::device_vector<float> m_sh2_second;
  thrust::device_vector<float> m_sh3_second;
};

}  // namespace tinygs
