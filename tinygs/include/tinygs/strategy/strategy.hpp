#pragma once

#include "tinygs/core/gpu_gaussian.hpp"
#include "tinygs/rasterizer/rasterizer.hpp"
namespace tinygs {

struct StrategyParams {
  // prune transparent gaussians
  float pruning_opacity_threshold = 0.005f;
  // prune large gaussians in world space
  float pruning_scale_threshold = 0.1f;
  // prune large gaussians in view space (2D)
  int max_screen_size = 20;

  // grow if gradient is large (Default Strategy)
  float duplicate_grad_threshold = 0.0002f;
  // split if large gaussian is found (Default Strategy)
  float duplicate_scale_threshold = 0.01f;

  int refine_every = 100;
  int start_refine = 1000;
  int end_refine = 15'000;
  int max_num_gaussians = 10'000'000;
  int reset_every = 3'000;
};

class StrategyBase {
public:
  explicit StrategyBase(std::shared_ptr<GPUGaussian3d> gaussians) : m_gaussians(gaussians) {}
  virtual ~StrategyBase() = default;

  // might use the densification info to densify the gaussians
  void step(const RasterizeContext& ctx);

  virtual void reset() = 0;

  using RemoveCallback = std::function<void(char* /*kept_flag*/, int /*num_kept*/)>;
  using DuplicateCallback = std::function<void(int* /*indices*/, int* /*new_indices*/, int /* num_duplications */)>;
  using ResetCallback = std::function<void(int* indices, int num_reset)>;

  virtual void step_impl(const RasterizeContext& ctx) = 0;
  void set_remove_callback(RemoveCallback callback) { m_remove_callback = callback; }
  void set_duplicate_callback(DuplicateCallback callback) { m_duplicate_callback = callback; }
  void set_reset_callback(ResetCallback callback) { m_reset_callback = callback; }

protected:
  void on_remove(char* kept_flag, int num_kept);
  void on_duplicate(int* indices, int* new_indices, int num_duplications);
  void on_reset(int* indices, int num_reset);

  int this_step() const noexcept { return m_step_count; }

  std::shared_ptr<GPUGaussian3d> m_gaussians;
  StrategyParams m_params;
private:
  int m_step_count = 0;

  RemoveCallback m_remove_callback;        /// use this function to remove some gaussians
  DuplicateCallback m_duplicate_callback;  /// use this function to duplicate some gaussians
  ResetCallback m_reset_callback;          /// use this function to reset some gaussians

};


}  // namespace tinygs