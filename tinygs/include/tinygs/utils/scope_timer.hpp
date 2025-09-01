#pragma once
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <limits>
#include <algorithm>

namespace tinygs {

class GlobalTimerRegistry {
public:
  struct TimerStats {
    size_t count = 0;
    std::vector<double> times;
    double total_time = 0.0;
    double min_time = std::numeric_limits<double>::max();
    double max_time = 0.0;
    
    void add_time(double time_ms) {
      times.push_back(time_ms);
      total_time += time_ms;
      count++;
      min_time = std::min(min_time, time_ms);
      max_time = std::max(max_time, time_ms);
    }
    
    double average_time() const {
      return count > 0 ? total_time / count : 0.0;
    }
  };

  static GlobalTimerRegistry& get_instance();
  
  void record_time(const std::string& name, double time_ms);
  const TimerStats* get_stats(const std::string& name) const;
  void print_all_stats() const;
  void clear();

private:
  GlobalTimerRegistry() = default;
  
  mutable std::mutex mutex_;
  std::map<std::string, TimerStats> timers_;
};

class ScopeTimer {
public:
  explicit ScopeTimer(const std::string& name)
    : m_name(name), m_start_time(std::chrono::high_resolution_clock::now()) {}
  
  ~ScopeTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_start_time);
    double time_ms = duration.count() / 1000.0;
    GlobalTimerRegistry::get_instance().record_time(m_name, time_ms);
  }
  
  // Non-copyable, non-movable for safety
  ScopeTimer(const ScopeTimer&) = delete;
  ScopeTimer& operator=(const ScopeTimer&) = delete;
  ScopeTimer(ScopeTimer&&) = delete;
  ScopeTimer& operator=(ScopeTimer&&) = delete;

private:
  std::string m_name;
  std::chrono::high_resolution_clock::time_point m_start_time;
};

// Convenience macro for easy usage
#define TINYGS_TIMER(name) tinygs::ScopeTimer timer_##__LINE__(name)

}  // namespace tinygs
