#pragma once

#include <memory>

#include "tinygs/platform/backend_types.hpp"

namespace tinygs {

class BackendContext {
public:
  virtual ~BackendContext() = default;

  virtual BackendType type() const noexcept = 0;
  virtual int device() const noexcept = 0;
};

class CudaBackendContext final : public BackendContext {
public:
  explicit CudaBackendContext(int device) : m_device(device) {}
  ~CudaBackendContext() override = default;

  BackendType type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return m_device; }

private:
  int m_device = 0;
};

}  // namespace tinygs
