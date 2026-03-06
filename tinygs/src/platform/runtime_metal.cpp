#include "tinygs/platform/runtime_factory.hpp"

#include <memory>
#include <utility>

namespace tinygs {

namespace {

class MetalQueue final : public BackendQueue {
public:
  explicit MetalQueue(int device) : m_device(device) {}

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override { return nullptr; }

private:
  int m_device = 0;
};

class MetalEvent final : public BackendEvent {
public:
  explicit MetalEvent(int device) : m_device(device) {}

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override { return nullptr; }

private:
  int m_device = 0;
};

class MetalBuffer final : public BackendBuffer {
public:
  MetalBuffer(int device, BufferDesc desc) : m_device(device), m_desc(std::move(desc)) {}

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  size_t size_bytes() const noexcept override { return m_desc.size_bytes; }
  const BufferDesc& desc() const noexcept override { return m_desc; }
  void* data() const noexcept override { return nullptr; }
  void* native_handle() const noexcept override { return nullptr; }

private:
  int m_device = 0;
  BufferDesc m_desc{};
};

class MetalRuntime final : public BackendRuntime {
public:
  explicit MetalRuntime(int device) : m_device(device) {}

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }

  CapabilityProfile capability_profile() const override {
    CapabilityProfile caps;
    caps.supports_queues = true;
    caps.supports_events = true;
    caps.supports_device_buffers = true;
    return caps;
  }

  Result<BackendQueue> create_queue(const QueueDesc&) override {
    return Result<BackendQueue>::success(std::make_shared<MetalQueue>(m_device), backend_type(), "create_queue");
  }

  Result<BackendEvent> create_event(const EventDesc&) override {
    return Result<BackendEvent>::success(std::make_shared<MetalEvent>(m_device), backend_type(), "create_event");
  }

  Result<BackendBuffer> create_buffer(const BufferDesc& desc) override {
    return Result<BackendBuffer>::success(std::make_shared<MetalBuffer>(m_device, desc), backend_type(), "create_buffer");
  }

  BackendError record_event(const std::shared_ptr<BackendQueue>&,
                            const std::shared_ptr<BackendEvent>&) override {
    return backend_success(backend_type(), "record_event");
  }

  BackendError wait_event(const std::shared_ptr<BackendQueue>&,
                          const std::shared_ptr<BackendEvent>&) override {
    return backend_success(backend_type(), "wait_event");
  }

  BackendError synchronize_queue(const std::shared_ptr<BackendQueue>&) override {
    return backend_success(backend_type(), "synchronize_queue");
  }

  BackendError synchronize_event(const std::shared_ptr<BackendEvent>&) override {
    return backend_success(backend_type(), "synchronize_event");
  }

  BackendError synchronize_device() override {
    return backend_success(backend_type(), "synchronize_device");
  }

  BackendError copy_buffer_async(const std::shared_ptr<BackendQueue>&,
                                 const std::shared_ptr<BackendBuffer>&,
                                 const std::shared_ptr<BackendBuffer>&,
                                 const CopyRegion&) override {
    return unsupported("copy_buffer_async");
  }

  BackendError copy_from_host_async(const std::shared_ptr<BackendQueue>&,
                                    const std::shared_ptr<BackendBuffer>&,
                                    const void*,
                                    const BufferTransferRegion&) override {
    return unsupported("copy_from_host_async");
  }

  BackendError copy_to_host_async(const std::shared_ptr<BackendQueue>&,
                                  void*,
                                  const std::shared_ptr<BackendBuffer>&,
                                  const BufferTransferRegion&) override {
    return unsupported("copy_to_host_async");
  }

  BackendError copy_device_to_host_async(const std::shared_ptr<BackendQueue>&,
                                         void*,
                                         const void*,
                                         size_t) override {
    return unsupported("copy_device_to_host_async");
  }

  BackendError copy_host_to_device_async(const std::shared_ptr<BackendQueue>&,
                                         void*,
                                         const void*,
                                         size_t) override {
    return unsupported("copy_host_to_device_async");
  }

  BackendError fill_buffer_async(const std::shared_ptr<BackendQueue>&,
                                 const std::shared_ptr<BackendBuffer>&,
                                 uint8_t,
                                 size_t,
                                 size_t) override {
    return unsupported("fill_buffer_async");
  }

private:
  BackendError unsupported(const char* operation) const {
    return backend_error(backend_type(),
                         BackendErrorCode::Unsupported,
                         operation,
                         "Metal runtime placeholder compiled successfully, but this operation is not implemented yet.");
  }

  int m_device = 0;
};

}  // namespace

Result<BackendRuntime> create_metal_backend_runtime(int device) {
  return Result<BackendRuntime>::success(std::make_shared<MetalRuntime>(device),
                                         BackendType::Metal,
                                         "create_metal_backend_runtime");
}

}  // namespace tinygs