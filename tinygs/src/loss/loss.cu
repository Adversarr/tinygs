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

LossBase::LossBase(std::shared_ptr<BackendRuntime> runtime)
    : m_runtime(std::move(runtime)) {}

LossBase::~LossBase() = default;

std::shared_ptr<BackendRuntime> LossBase::runtime() const {
  return m_runtime;
}

MetricBase::MetricBase(std::shared_ptr<BackendRuntime> runtime)
    : m_runtime(std::move(runtime)) {}

MetricBase::~MetricBase() = default;

std::shared_ptr<BackendRuntime> MetricBase::runtime() const {
  return m_runtime;
}

std::unique_ptr<LossBase> create_loss(std::shared_ptr<BackendRuntime> runtime, const std::string& loss_type) {
  std::string lower_loss_type = to_lower(loss_type);
  
  if (lower_loss_type == "l1") {
    return std::make_unique<L1Loss>(std::move(runtime));
  } else if (lower_loss_type == "l2") {
    return std::make_unique<L2Loss>(std::move(runtime));
  } else if (lower_loss_type == "huber") {
    return std::make_unique<HuberLoss>(std::move(runtime));
  } else if (lower_loss_type == "fused_ssim") {
    return std::make_unique<FusedSSIMLoss>(std::move(runtime));
  } else {
    throw std::runtime_error("Unknown loss type: " + loss_type);
  }
}

std::unique_ptr<MetricBase> create_metric(std::shared_ptr<BackendRuntime> runtime, const std::string& metric_type) {
  std::string lower_metric_type = to_lower(metric_type);
  
  if (lower_metric_type == "psnr") {
    return std::make_unique<PsnrMetric>(std::move(runtime));
  } else {
    throw std::runtime_error("Unknown metric type: " + metric_type);
  }
}


}  // namespace tinygs
