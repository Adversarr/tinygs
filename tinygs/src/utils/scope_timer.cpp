#include "tinygs/utils/scope_timer.hpp"
#include <iostream>
#include <iomanip>

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
    std::cout << "No timer statistics available.\n";
    return;
  }
  
  int expected_max_func_name_length = 50;
  std::cout << "\n=== Timer Statistics ===\n";
  std::cout << std::left << std::setw(expected_max_func_name_length) << "Timer Name" 
            << std::setw(10) << "Count"
            << std::setw(12) << "Total (ms)"
            << std::setw(12) << "Avg (ms)"
            << std::setw(12) << "Min (ms)"
            << std::setw(12) << "Max (ms)" << "\n";
  std::cout << std::string(expected_max_func_name_length + 53, '-') << "\n";
  
  for (const auto& [name, stats] : timers_) {
    std::cout << std::left << std::setw(expected_max_func_name_length) << name.substr(0, expected_max_func_name_length)
              << std::setw(10) << stats.count
              << std::setw(12) << std::fixed << std::setprecision(3) << stats.total_time
              << std::setw(12) << std::fixed << std::setprecision(3) << stats.average_time()
              << std::setw(12) << std::fixed << std::setprecision(3) << stats.min_time
              << std::setw(12) << std::fixed << std::setprecision(3) << stats.max_time << "\n";
  }
  std::cout << "\n";
}

void GlobalTimerRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  timers_.clear();
}

}  // namespace tinygs