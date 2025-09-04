#pragma once

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
namespace tinygs {

struct StrategyParams {
  float pruning_opacity_threshold = 0.005f;
  float pruning_scale_threshold = 0.1f;
  float duplicate_grad_threshold = 0.0002f;
  float duplicate_scale_threshold = 0.01f;
  int refine_every = 100;
  int start_refine = 15'000;
  int max_num_gaussians = 10'000'000;
};

class StrategyBase {
public:
  explicit StrategyBase(std::shared_ptr<GPUGaussian3d> gaussians) : m_gaussians(gaussians) {}
  virtual ~StrategyBase() = default;

  // might use the densification info to densify the gaussians
  virtual void step(const RasterizeContext& ctx) = 0;

  virtual void reset() = 0;


  using RemoveCallback = std::function<void(char* /*kept_flag*/, int /*num_kept*/)>;
  using DuplicateCallback = std::function<void(int* /*indices*/, int* /*new_indices*/, int /* num_duplications */)>;

  void set_pre_remove_callback(RemoveCallback callback) { m_remove_callback = callback; }
  void set_post_duplicate_callback(DuplicateCallback callback) { m_duplicate_callback = callback; }

protected:
  void remove(char* kept_flag, int num_kept) {
    if (m_remove_callback) {
      m_remove_callback(kept_flag, num_kept);
    }
  }

  void post_duplicate(int* indices, int* new_indices, int num_duplications) {
    if (m_duplicate_callback) {
      m_duplicate_callback(indices, new_indices, num_duplications);
    }
  }
  std::shared_ptr<GPUGaussian3d> m_gaussians;
  StrategyParams m_params;

private:
  RemoveCallback m_remove_callback;        /// use this function to remove some gaussians
  DuplicateCallback m_duplicate_callback;  /// use this function to duplicate some gaussians
};

}  // namespace tinygs