#pragma once
#include "tinygs/cuda/gpu_memory.hpp"
#include "tinygs/optim/optim.hpp"

namespace tinygs {

struct SGDParameters {
  // No momentum parameters needed for basic SGD
};

/**
 * @brief Basic SGD optimizer without momentum for Gaussian Splatting
 *
 */
class SGD : public OptimizerBase {
public:
  SGD(std::shared_ptr<GPUGaussian3d> gaussians, std::shared_ptr<GPUGaussian3d> gaussians_grad,
        const SGDParameters& params = {});

  ~SGD() override = default;

  void reset() override;
  void step(float scale) override;
  void remove(char* kept_flag, int num_kept) override;
  void duplicate(int* indices, int* new_indices, int num_duplicate) override;
  void reset(int* indices, int num_reset) override;

private:
  SGDParameters m_sgd_params;
};

}  // namespace tinygs