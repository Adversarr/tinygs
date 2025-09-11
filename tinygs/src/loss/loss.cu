#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "tinygs/loss/loss.hpp"
#include "tinygs/loss/l1.hpp"
#include "tinygs/loss/fused_ssim.hpp"
#include "tinygs/loss/psnr.hpp"
#include <algorithm>

namespace tinygs {

std::unique_ptr<LossBase> create_loss(const std::string& loss_type) {
  std::string lower_loss_type = loss_type;
  std::transform(lower_loss_type.begin(), lower_loss_type.end(), lower_loss_type.begin(), ::tolower);
  
  if (lower_loss_type == "l1") {
    return std::make_unique<L1Loss>();
  } else if (lower_loss_type == "fused_ssim") {
    return std::make_unique<FusedSSIMLoss>();
  } else {
    throw std::runtime_error("Unknown loss type: " + loss_type);
  }
}

std::unique_ptr<MetricBase> create_metric(const std::string& metric_type) {
  std::string lower_metric_type = metric_type;
  std::transform(lower_metric_type.begin(), lower_metric_type.end(), lower_metric_type.begin(), ::tolower);
  
  if (lower_metric_type == "psnr") {
    return std::make_unique<PsnrMetric>();
  } else {
    throw std::runtime_error("Unknown metric type: " + metric_type);
  }
}


}  // namespace tinygs
