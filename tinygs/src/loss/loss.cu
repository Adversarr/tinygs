#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "tinygs/loss/loss.hpp"
#include "tinygs/loss/l1.hpp"
#include "tinygs/loss/l2.hpp"
#include "tinygs/loss/huber.hpp"
#include "tinygs/loss/fused_ssim.hpp"
#include "tinygs/loss/psnr.hpp"
#include <algorithm>
#include <memory>

namespace tinygs {

LossBase::LossBase(BackendRuntime& runtime)
    : m_runtime(&runtime) {}

LossBase::~LossBase() = default;

BackendRuntime& LossBase::runtime() const {
  return *m_runtime;
}

MetricBase::MetricBase(BackendRuntime& runtime)
    : m_runtime(&runtime) {}

MetricBase::~MetricBase() = default;

BackendRuntime& MetricBase::runtime() const {
  return *m_runtime;
}

std::unique_ptr<LossBase> create_loss(BackendRuntime& runtime, const std::string& loss_type) {
  std::string lower_loss_type = to_lower(loss_type);
  
  if (lower_loss_type == "l1") {
    return std::make_unique<L1Loss>(runtime);
  } else if (lower_loss_type == "l2") {
    return std::make_unique<L2Loss>(runtime);
  } else if (lower_loss_type == "huber") {
    return std::make_unique<HuberLoss>(runtime);
  } else if (lower_loss_type == "fused_ssim") {
    return std::make_unique<FusedSSIMLoss>(runtime);
  } else {
    throw std::runtime_error("Unknown loss type: " + loss_type);
  }
}

std::unique_ptr<MetricBase> create_metric(BackendRuntime& runtime, const std::string& metric_type) {
  std::string lower_metric_type = to_lower(metric_type);
  
  if (lower_metric_type == "psnr") {
    return std::make_unique<PsnrMetric>(runtime);
  } else {
    throw std::runtime_error("Unknown metric type: " + metric_type);
  }
}


}  // namespace tinygs
