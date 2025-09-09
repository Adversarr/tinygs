#include "tinygs/strategy/strategy.hpp"

namespace tinygs {

void StrategyBase::step(const RasterizeContext& ctx) {
  step_impl(ctx);
  ++m_step_count;
}

void StrategyBase::on_remove(char* kept_flag, int num_kept) {
  if (num_kept <= 0) return;
  
  if (m_gaussians) {
    m_gaussians->remove(kept_flag, num_kept);
  }
  if (m_gaussians_grad) {
    m_gaussians_grad->remove(kept_flag, num_kept);
  }
  if (m_optimizer) {
    m_optimizer->remove(kept_flag, num_kept);
  }
}

void StrategyBase::on_duplicate(int* indices, int* new_indices, int num_duplications) {
  if (num_duplications <= 0) return;
  
  if (m_gaussians) {
    m_gaussians->append(num_duplications);
  }
  if (m_gaussians_grad) {
    m_gaussians_grad->append(num_duplications);
  }
  if (m_optimizer) {
    m_optimizer->duplicate(indices, new_indices, num_duplications);
  }
}

void StrategyBase::on_reset(int* indices, int num_reset) {
  if (num_reset <= 0) return;
  
  if (m_optimizer) {
    m_optimizer->reset(indices, num_reset);
  }
}

void StrategyBase::on_reset_opacity() {
  if (m_optimizer) {
    m_optimizer->reset_opacity();
  }
}


}