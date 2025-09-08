#include "tinygs/strategy/strategy.hpp"

namespace tinygs {

void StrategyBase::step(const RasterizeContext& ctx) {
  step_impl(ctx);
  ++m_step_count;
}

void StrategyBase::on_remove(char* kept_flag, int num_kept) {
  if (m_remove_callback) {
    m_remove_callback(kept_flag, num_kept);
  } else {
    log_warning("No remove callback set!");
  }
}

void StrategyBase::on_duplicate(int* indices, int* new_indices, int num_duplications) {
  if (m_duplicate_callback) {
    m_duplicate_callback(indices, new_indices, num_duplications);
  } else {
    log_warning("No duplicate callback set!");
  }
}

void StrategyBase::on_reset(int* indices, int num_reset) {
  if (m_reset_callback) {
    m_reset_callback(indices, num_reset);
  } else {
    log_warning("No reset callback set!");
  }
}

void StrategyBase::on_reset_opacity() {
  if (m_reset_opacity_callback) {
    m_reset_opacity_callback();
  } else {
    log_warning("No reset opacity callback set!");
  }
}


}