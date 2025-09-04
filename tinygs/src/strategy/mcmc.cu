#include "tinygs/strategy/mcmc.hpp"
#include "tinygs/random/multinomial.hpp"
#include "tinygs/cuda/common_device.cuh"

namespace tinygs {

MCMCStrategy::MCMCStrategy(std::shared_ptr<GPUGaussian3d> gaussians) : StrategyBase(gaussians) {
}


void MCMCStrategy::step(const RasterizeContext& ctx) {
  // TODO: disable some functionalities if total step is large
  // pruning(ctx);
  // add_new_gs(ctx);
  // add_noise(ctx);
}

void MCMCStrategy::reset() {
  // TODO: reset internal states
}

void MCMCStrategy::set_noise_lr(float noise_lr) { m_noise_lr = noise_lr; }

MCMCStrategy::~MCMCStrategy() = default;



void MCMCStrategy::pruning(const RasterizeContext& /* ctx */) {
  // Remove dead gaussians
  const auto num_gaussians = m_gaussians->size();
  GPUBuffer<char> is_alive(num_gaussians);
  const auto* d_opacity = thrust::raw_pointer_cast(m_gaussians->opacities().data());
  const auto* d_rotations = thrust::raw_pointer_cast(m_gaussians->rotations().data());

  thrust::for_each(                                                          //
      thrust::make_counting_iterator<int>(0),                                //
      thrust::make_counting_iterator<int>(num_gaussians),                    //
      [d_is_alive = is_alive.data(), d_opacity, d_rotations,                 //
       min_opacity = m_params.pruning_opacity_threshold] __device__(int i) { //
        if (d_opacity[i] > min_opacity && length2(d_rotations[i]) > 1.0e-8f) {
          d_is_alive[i] = 1;
        } else {
          d_is_alive[i] = 0;
        }
      });
  int nums_kept = thrust::reduce( //
      thrust::device_ptr<char>(is_alive.data()),
      thrust::device_ptr<char>(is_alive.data() + num_gaussians), 0,
      thrust::plus<char>());

  this->remove(is_alive.data(), nums_kept);
  log_info("Remove {} dead gaussians", num_gaussians - nums_kept);
}

} // namespace tinygs
