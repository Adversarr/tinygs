#pragma once
#include "cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"

namespace tinygs {

struct AdamWParameters {
  // Shared parameters
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;

  /// NOTE: from tiny-cuda-nn Adam implementation
  // AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
  bool enable_adabound = false;
};

/**
 * @brief A baseline version for Gaussian Splatting
 *
 */
class AdamW : public OptimizerBase {
public:
  AdamW(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad,
        const AdamWParameters& params = {});

  ~AdamW() override = default;

  void reset() override;
  void step(float scale) override;
  void pre_remove(char* kept_flag) override;
  void post_duplicate(int* indices, int* new_indices, int num_duplicate) override;
private:
  AdamWParameters m_adam_params;

  // TODO: replace with ours.
  thrust::device_vector<vec3> m_means_first_second;
  thrust::device_vector<float> m_opacities_first_second;
  thrust::device_vector<vec4> m_rotations_first_second;
  thrust::device_vector<vec3> m_scales_first_second;
  thrust::device_vector<vec3> m_sh_coefficients_first_second;
  thrust::device_vector<uint32_t> m_gaussian_steps;
};

}  // namespace tinygs
