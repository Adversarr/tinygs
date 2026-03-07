// Metal Backend Runtime Implementation (NVI)
//
// Implements the protected do_* primitives of BackendRuntime for Apple Metal.
// All argument validation (nulls, bounds, type checks) is handled by the NVI
// base class; this file contains only native Metal API calls.
//
// Memory Model:
// - All buffers use MTLResourceStorageModeShared (unified memory)
// - Host copies are direct memcpy (zero-copy on Apple Silicon)
// - GPU copies use blit command encoders
//
// Synchronization:
// - Events use MTLSharedEvent with monotonic signal values
// - synchronize_event uses dispatch_semaphore + notifyListener for blocking CPU wait
// - synchronize_device uses base class default (sync all live queues)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "tinygs/platform/runtime.hpp"
#include "tinygs/common.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace tinygs {

namespace {

// ============================================================================
// Metal resource wrappers
// ============================================================================

class MetalQueue final : public BackendQueue {
public:
  MetalQueue(int device, id<MTLCommandQueue> queue)
      : m_device(device), m_queue(queue) {}

  ~MetalQueue() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override { return (__bridge void*)m_queue; }

  id<MTLCommandQueue> queue() const noexcept { return m_queue; }

private:
  int m_device = 0;
  id<MTLCommandQueue> m_queue = nullptr;
};

class MetalEvent final : public BackendEvent {
public:
  MetalEvent(int device, id<MTLSharedEvent> event)
      : m_device(device), m_event(event), m_signaled_value(0) {}

  ~MetalEvent() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override { return (__bridge void*)m_event; }

  id<MTLSharedEvent> event() const noexcept { return m_event; }

  uint64_t get_and_increment_signal_value() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return ++m_signaled_value;
  }

  uint64_t current_signal_value() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_signaled_value;
  }

private:
  int m_device = 0;
  id<MTLSharedEvent> m_event = nullptr;
  uint64_t m_signaled_value = 0;
  mutable std::mutex m_mutex;
};

class MetalBuffer final : public BackendBuffer {
public:
  MetalBuffer(int device, BufferDesc desc, id<MTLBuffer> buffer)
      : m_device(device), m_desc(std::move(desc)), m_buffer(buffer) {}

  ~MetalBuffer() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  size_t size_bytes() const noexcept override { return m_desc.size_bytes; }
  const BufferDesc& desc() const noexcept override { return m_desc; }
  void* data() const noexcept override { return m_buffer.contents; }
  void* native_handle() const noexcept override { return (__bridge void*)m_buffer; }

  id<MTLBuffer> buffer() const noexcept { return m_buffer; }

private:
  int m_device = 0;
  BufferDesc m_desc;
  id<MTLBuffer> m_buffer = nullptr;
};

// ============================================================================
// MetalRuntime — do_* primitives only
// ============================================================================

class MetalRuntime final : public BackendRuntime {
public:
  explicit MetalRuntime(int device) : m_device(device) {
    @autoreleasepool {
      if (m_device < 0) {
        throw std::runtime_error("backend.device must be >= 0 for Metal runtime.");
      }

      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      if (m_device >= static_cast<int>([devices count])) {
        throw std::runtime_error("Requested Metal device index out of range.");
      }

      m_mtl_device = devices[m_device];

      m_capability_profile.supports_queues = true;
      m_capability_profile.supports_events = true;
      m_capability_profile.supports_device_buffers = true;

      if ([m_mtl_device respondsToSelector:@selector(hasUnifiedMemory)]) {
        m_capability_profile.supports_unified_memory = m_mtl_device.hasUnifiedMemory;
        m_capability_profile.supports_host_visible_buffers = m_mtl_device.hasUnifiedMemory;
      } else {
        m_capability_profile.supports_unified_memory = true;
        m_capability_profile.supports_host_visible_buffers = true;
      }

      m_capability_profile.supports_interop_buffers = false;
      m_capability_profile.supports_graph_capture = false;

      if ([m_mtl_device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        m_capability_profile.total_global_memory_bytes =
            static_cast<size_t>(m_mtl_device.recommendedMaxWorkingSetSize);
      }
    }
  }

  ~MetalRuntime() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  CapabilityProfile capability_profile() const override { return m_capability_profile; }

protected:
  BackendError wait_for_queue_completion(MetalQueue& queue, const char* op) {
    id<MTLCommandBuffer> sync_cmd = [queue.queue() commandBuffer];
    if (!sync_cmd) {
      return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                           op, "Failed to create command buffer for queue synchronization");
    }
    [sync_cmd commit];
    [sync_cmd waitUntilCompleted];
    return backend_success(BackendType::Metal, op);
  }

  // ---- Resource creation ----

  Result<BackendQueue> do_create_queue(const QueueDesc& /*desc*/) override {
    @autoreleasepool {
      constexpr const char* op = "create_queue";
      id<MTLCommandQueue> queue = [m_mtl_device newCommandQueue];
      if (!queue) {
        return Result<BackendQueue>::failure(
            backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                          op, "Failed to create Metal command queue"));
      }
      return Result<BackendQueue>::success(
          std::make_shared<MetalQueue>(m_device, queue), BackendType::Metal, op);
    }
  }

  Result<BackendEvent> do_create_event(const EventDesc& /*desc*/) override {
    @autoreleasepool {
      constexpr const char* op = "create_event";
      id<MTLSharedEvent> event = [m_mtl_device newSharedEvent];
      if (!event) {
        return Result<BackendEvent>::failure(
            backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                          op, "Failed to create Metal shared event"));
      }
      return Result<BackendEvent>::success(
          std::make_shared<MetalEvent>(m_device, event), BackendType::Metal, op);
    }
  }

  Result<BackendBuffer> do_create_buffer(const BufferDesc& desc) override {
    @autoreleasepool {
      constexpr const char* op = "create_buffer";
      MTLResourceOptions options = MTLResourceStorageModeShared;
      id<MTLBuffer> buffer = [m_mtl_device newBufferWithLength:desc.size_bytes
                                                       options:options];
      if (!buffer) {
        return Result<BackendBuffer>::failure(
            backend_error(BackendType::Metal, BackendErrorCode::OutOfMemory,
                          op, "Metal buffer allocation failed"));
      }
      BufferDesc normalized_desc = desc;
      normalized_desc.host_access = BufferHostAccess::ReadWrite;
      return Result<BackendBuffer>::success(
          std::make_shared<MetalBuffer>(m_device, std::move(normalized_desc), buffer),
          BackendType::Metal, op);
    }
  }

  // ---- Synchronization primitives ----

  BackendError do_record_event(BackendQueue& queue, BackendEvent& event) override {
    @autoreleasepool {
      auto& mq = static_cast<MetalQueue&>(queue);
      auto& me = static_cast<MetalEvent&>(event);
      id<MTLCommandBuffer> cmd = [mq.queue() commandBuffer];
      if (!cmd) {
        return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                             "record_event", "Failed to create command buffer");
      }
      uint64_t signal_value = me.get_and_increment_signal_value();
      [cmd encodeSignalEvent:me.event() value:signal_value];
      [cmd commit];
      return backend_success(BackendType::Metal, "record_event");
    }
  }

  BackendError do_wait_event(BackendQueue& queue, BackendEvent& event) override {
    @autoreleasepool {
      auto& mq = static_cast<MetalQueue&>(queue);
      auto& me = static_cast<MetalEvent&>(event);
      id<MTLCommandBuffer> cmd = [mq.queue() commandBuffer];
      if (!cmd) {
        return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                             "wait_event", "Failed to create command buffer");
      }
      uint64_t wait_value = me.current_signal_value();
      [cmd encodeWaitForEvent:me.event() value:wait_value];
      [cmd commit];
      return backend_success(BackendType::Metal, "wait_event");
    }
  }

  BackendError do_synchronize_queue(BackendQueue& queue) override {
    @autoreleasepool {
      auto& mq = static_cast<MetalQueue&>(queue);
      id<MTLCommandBuffer> cmd = [mq.queue() commandBuffer];
      if (!cmd) {
        return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                             "synchronize_queue", "Failed to create command buffer");
      }
      [cmd commit];
      [cmd waitUntilCompleted];
      return backend_success(BackendType::Metal, "synchronize_queue");
    }
  }

  BackendError do_synchronize_event(BackendEvent& event) override {
    @autoreleasepool {
      auto& me = static_cast<MetalEvent&>(event);
      uint64_t target = me.current_signal_value();
      // Fast path: already signaled.
      if (me.event().signaledValue >= target) {
        return backend_success(BackendType::Metal, "synchronize_event");
      }
      // Block CPU until the event reaches the target value.
      dispatch_semaphore_t sem = dispatch_semaphore_create(0);
      MTLSharedEventListener* listener = [[MTLSharedEventListener alloc] init];
      [me.event() notifyListener:listener
                         atValue:target
                           block:^(id<MTLSharedEvent> /*e*/, uint64_t /*v*/) {
                             dispatch_semaphore_signal(sem);
                           }];
      dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
      return backend_success(BackendType::Metal, "synchronize_event");
    }
  }

  // synchronize_device: uses default base-class implementation (sync all live queues).

  // ---- Data transfer primitives ----

  BackendError do_copy_buffer(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    @autoreleasepool {
      auto& mq = static_cast<MetalQueue&>(queue);
      auto& md = static_cast<MetalBuffer&>(dst);
      auto& ms = static_cast<MetalBuffer&>(src);
      id<MTLCommandBuffer> cmd = [mq.queue() commandBuffer];
      if (!cmd) {
        return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                             "copy_buffer_async", "Failed to create command buffer");
      }
      id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
      if (!enc) {
        return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                             "copy_buffer_async", "Failed to create blit command encoder");
      }
      [enc copyFromBuffer:ms.buffer() sourceOffset:src_offset
                 toBuffer:md.buffer() destinationOffset:dst_offset
                     size:size_bytes];
      [enc endEncoding];
      [cmd commit];
      return backend_success(BackendType::Metal, "copy_buffer_async");
    }
  }

  BackendError do_copy_from_host(
      BackendQueue& queue,
      BackendBuffer& dst, size_t dst_offset,
      const void* src, size_t size_bytes) override {
    auto& mq = static_cast<MetalQueue&>(queue);
    auto sync_status = wait_for_queue_completion(mq, "copy_from_host_async");
    if (!sync_status.ok()) {
      return sync_status;
    }
    void* dst_ptr = static_cast<char*>(dst.data()) + dst_offset;
    std::memcpy(dst_ptr, src, size_bytes);
    return backend_success(BackendType::Metal, "copy_from_host_async");
  }

  BackendError do_copy_to_host(
      BackendQueue& queue,
      void* dst,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    auto& mq = static_cast<MetalQueue&>(queue);
    auto sync_status = wait_for_queue_completion(mq, "copy_to_host_async");
    if (!sync_status.ok()) {
      return sync_status;
    }
    const void* src_ptr = static_cast<const char*>(src.data()) + src_offset;
    std::memcpy(dst, src_ptr, size_bytes);
    return backend_success(BackendType::Metal, "copy_to_host_async");
  }

  BackendError do_transfer_raw(
      BackendQueue& queue,
      void* dst, const void* src,
      size_t size_bytes,
      TransferDirection /*direction*/) override {
    auto& mq = static_cast<MetalQueue&>(queue);
    auto sync_status = wait_for_queue_completion(mq, "transfer_raw");
    if (!sync_status.ok()) {
      return sync_status;
    }
    std::memcpy(dst, src, size_bytes);
    return backend_success(BackendType::Metal, "transfer_raw");
  }

  BackendError do_fill_buffer(
      BackendQueue& queue,
      BackendBuffer& buffer,
      size_t offset, uint8_t value, size_t size_bytes) override {
    @autoreleasepool {
      auto& mq = static_cast<MetalQueue&>(queue);
      auto& mb = static_cast<MetalBuffer&>(buffer);
      id<MTLCommandBuffer> cmd = [mq.queue() commandBuffer];
      if (!cmd) {
        return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                             "fill_buffer_async", "Failed to create command buffer");
      }
      id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
      if (!enc) {
        return backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                             "fill_buffer_async", "Failed to create blit command encoder");
      }
      NSRange range = NSMakeRange(offset, size_bytes);
      [enc fillBuffer:mb.buffer() range:range value:value];
      [enc endEncoding];
      [cmd commit];
      return backend_success(BackendType::Metal, "fill_buffer_async");
    }
  }

private:
  int m_device = 0;
  id<MTLDevice> m_mtl_device = nullptr;
  CapabilityProfile m_capability_profile;
};

}  // namespace

Result<BackendRuntime> create_metal_backend_runtime(int device) {
  constexpr const char* operation = "create_metal_backend_runtime";
  @autoreleasepool {
    @try {
      return Result<BackendRuntime>::success(
          std::make_shared<MetalRuntime>(device), BackendType::Metal, operation);
    } @catch (NSException* exception) {
      NSString* reason = exception.reason ?: @"Unknown Metal initialization error";
      return Result<BackendRuntime>::failure(
          backend_error(BackendType::Metal, BackendErrorCode::RuntimeFailure,
                        operation, [reason UTF8String]));
    }
  }
}

}  // namespace tinygs
