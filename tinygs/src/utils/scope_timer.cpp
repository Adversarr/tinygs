#include "tinygs/utils/scope_timer.hpp"
#include "tinygs/cuda/common_host.hpp"

namespace tinygs {

GlobalTimerRegistry& GlobalTimerRegistry::get_instance() {
  static GlobalTimerRegistry instance;
  return instance;
}

void GlobalTimerRegistry::record_time(const std::string& name, double time_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  timers_[name].add_time(time_ms);
}

const GlobalTimerRegistry::TimerStats* GlobalTimerRegistry::get_stats(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = timers_.find(name);
  return (it != timers_.end()) ? &it->second : nullptr;
}

void GlobalTimerRegistry::print_all_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (timers_.empty()) {
    log_info("No timer statistics available.");
    return;
  }
  
  log_info("=== Timer Statistics ===");
  log_info("{:<50s} {:>8s} {:>12s} {:>12s} {:>12s} {:>12s}",
           "Timer Name", "Count", "Total (ms)", "Avg (ms)", "Min (ms)", "Max (ms)");
  
  for (const auto& [name, stats] : timers_) {
    log_info("{:<50s} {:>8d} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}",
             name.substr(0, 50), stats.count,
             stats.total_time, stats.average_time(),
             stats.min_time, stats.max_time);
  }
}

void GlobalTimerRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  timers_.clear();
}

}  // namespace tinygs