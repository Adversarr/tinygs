#pragma once
#include "tinygs/core/gpu_gaussian.hpp"
namespace tinygs {

class InspectChange {
public:
  explicit InspectChange(std::shared_ptr<GPUGaussian3d> gaussian) : m_gaussian(gaussian) {
    m_gaussian_old = m_gaussian->clone();
  }

  void step();
private:
  std::shared_ptr<GPUGaussian3d> m_gaussian;
  std::unique_ptr<GPUGaussian3d> m_gaussian_old;
};

}