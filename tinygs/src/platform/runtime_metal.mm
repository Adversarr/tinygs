// Metal Backend Runtime Implementation
//
// This file implements the backend-neutral runtime contract (BackendRuntime) using Apple's Metal
// framework. It provides a concrete implementation for GPU compute operations on macOS/iOS devices.
//
// Architecture Overview:
// - MetalQueue: Wraps MTLCommandQueue for command submission
// - MetalEvent: Wraps MTLSharedEvent for synchronization between command buffers
// - MetalBuffer: Wraps MTLBuffer for GPU-accessible memory
// - MetalRuntime: Main entry point implementing BackendRuntime interface
//
// Key Metal Concepts:
// - Command Queues: Serialize command buffer submission to the GPU
// - Shared Events: Cross-process and cross-queue synchronization primitives
// - Unified Memory: On Apple Silicon, CPU and GPU share the same memory space
// - Autorelease Pools: Required for proper Objective-C memory management in C++ context
//
// Implementation Notes:
// - All Metal allocations use MTLResourceStorageModeShared for unified memory access
// - Interop buffers (for multi-GPU scenarios) are not yet implemented
// - Graph capture (for optimization) is not yet supported
// - Synchronization is cooperative; Metal does not have implicit device-wide barriers

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tinygs/platform/runtime_contract.hpp"
#include "tinygs/common.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace tinygs {

namespace {

// ============================================================================
// Helper Functions
// ============================================================================

// Returns true if value is a power of two (required for buffer alignment)
inline bool is_power_of_two(size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

// Normalizes operation names for error reporting; prevents empty strings
inline std::string normalize_operation(const std::string& operation) {
  if (operation.empty()) {
    return "unspecified_operation";
  }
  return operation;
}

// Detects size_t overflow for safe bounds checking
inline bool add_overflows(size_t lhs, size_t rhs) noexcept {
  return lhs > std::numeric_limits<size_t>::max() - rhs;
}

// Converts NSError to BackendError; returns success if error is nil
BackendError metal_status(NSError* error, const std::string& operation) {
  const std::string op_name = normalize_operation(operation);
  if (!error) {
    return backend_success(BackendType::Metal, op_name);
  }
  
  NSString* nsDesc = error.localizedDescription ?: @"Unknown Metal error";
  std::string message = [nsDesc UTF8String];
  
  return backend_error(BackendType::Metal,
                       BackendErrorCode::RuntimeFailure,
                       op_name,
                       message);
}

// ============================================================================
// MetalQueue - Command Queue Wrapper
// ============================================================================
//
// Wraps MTLCommandQueue which serializes command buffer submission to a Metal device.
// Each queue maintains independent execution order; multiple queues can execute concurrently.
//
// Thread Safety:
// - MTLCommandQueue is thread-safe; multiple threads can create command buffers simultaneously
// - Command buffers from the same queue execute in submission order

class MetalQueue final : public BackendQueue {
public:
  MetalQueue(int device, id<MTLCommandQueue> queue)
      : m_device(device), m_queue(queue) {
  }

  ~MetalQueue() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override {
    return (__bridge void*)m_queue;
  }

  id<MTLCommandQueue> queue() const noexcept { return m_queue; }

private:
  int m_device = 0;
  id<MTLCommandQueue> m_queue = nullptr;
};

// ============================================================================
// MetalEvent - Synchronization Primitive Wrapper
// ============================================================================
//
// Wraps MTLSharedEvent for cross-queue and cross-process synchronization.
// Metal events use monotonically increasing values rather than binary signaled state.
//
// Usage Pattern:
// 1. Signal: encodeSignalEvent with value N (GPU will signal when command buffer completes)
// 2. Wait: encodeWaitForEvent with value N (GPU blocks until event reaches value N)
//
// Thread Safety:
// - m_signaled_value is protected by mutex for concurrent signal/wait operations
// - MTLSharedEvent itself is thread-safe

class MetalEvent final : public BackendEvent {
public:
  MetalEvent(int device, id<MTLSharedEvent> event)
      : m_device(device), m_event(event), m_signaled_value(0) {
  }

  ~MetalEvent() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  void* native_handle() const noexcept override {
    return (__bridge void*)m_event;
  }

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

// ============================================================================
// MetalBuffer - GPU Memory Buffer Wrapper
// ============================================================================
//
// Wraps MTLBuffer for GPU-accessible memory. Uses MTLResourceStorageModeShared which:
// - Allocates memory accessible to both CPU and GPU (unified memory model)
// - On Apple Silicon: physically shared memory, zero-copy access
// - On Intel Macs: system memory with coherent caching
//
// Buffer Contents:
// - data() returns direct pointer via buffer.contents (no explicit map/unmap required)
// - Memory is always host-visible and coherent; no flush/invalidate needed
// - Alignment: Metal guarantees 256-byte alignment by default

class MetalBuffer final : public BackendBuffer {
public:
  MetalBuffer(int device, BufferDesc desc, id<MTLBuffer> buffer)
      : m_device(device), m_desc(std::move(desc)), m_buffer(buffer) {
  }

  ~MetalBuffer() override = default;

  BackendType backend_type() const noexcept override { return BackendType::Metal; }
  int device() const noexcept override { return m_device; }
  size_t size_bytes() const noexcept override { return m_desc.size_bytes; }
  const BufferDesc& desc() const noexcept override { return m_desc; }
  void* data() const noexcept override {
    return m_buffer.contents;
  }
  void* native_handle() const noexcept override {
    return (__bridge void*)m_buffer;
  }

  id<MTLBuffer> buffer() const noexcept { return m_buffer; }

private:
  int m_device = 0;
  BufferDesc m_desc;
  id<MTLBuffer> m_buffer = nullptr;
};

// ============================================================================
// MetalRuntime - Main Backend Implementation
// ============================================================================
//
// Implements BackendRuntime for Metal, providing device management, resource allocation,
// and execution control. All operations are asynchronous and queue-based.
//
// Device Selection:
// - Device 0 is typically the integrated GPU (or Apple Silicon GPU)
// - Higher indices are external GPUs (eGPU) on Intel Macs
// - MTLCopyAllDevices() returns all available Metal devices
//
// Capability Detection:
// - Unified memory: Apple Silicon and late Intel Macs share CPU/GPU memory
// - recommendedMaxWorkingSetSize: suggested memory budget (not hard limit)
// - respondToSelector checks guard against older macOS versions
//
// Memory Model:
// - All buffers use shared storage mode (unified memory semantics)
// - No explicit staging buffers needed; CPU can directly read/write GPU memory
// - Interop with other GPUs/backends not yet implemented

class MetalRuntime final : public BackendRuntime {
public:
  explicit MetalRuntime(int device) : m_device(device) {
    @autoreleasepool {
      if (m_device < 0) {
        throw std::runtime_error("backend.device must be >= 0 for Metal runtime.");
      }

      NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
      
      if (m_device >= [devices count]) {
        throw std::runtime_error("Requested Metal device index out of range.");
      }
      
      m_mtl_device = devices[m_device];

      // Query device capabilities for runtime feature detection
      m_capability_profile.supports_queues = true;
      m_capability_profile.supports_events = true;
      m_capability_profile.supports_device_buffers = true;
      
      // Check for unified memory (Apple Silicon and modern Intel Macs)
      // hasUnifiedMemory API available on macOS 10.15+
      if ([m_mtl_device respondsToSelector:@selector(hasUnifiedMemory)]) {
        m_capability_profile.supports_unified_memory = m_mtl_device.hasUnifiedMemory;
        m_capability_profile.supports_host_visible_buffers = m_mtl_device.hasUnifiedMemory;
      } else {
        // Fallback: assume unified memory on older macOS versions
        m_capability_profile.supports_unified_memory = true;
        m_capability_profile.supports_host_visible_buffers = true;
      }
      
      // Interop and graph capture not yet implemented for Metal backend
      m_capability_profile.supports_interop_buffers = false;
      m_capability_profile.supports_graph_capture = false;
      
      // recommendedMaxWorkingSetSize available on macOS 10.12+
      // This is a soft limit; exceeding it may cause performance degradation
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

  // ============================================================================
  // Resource Creation
  // ============================================================================

  // Creates a new command queue for submitting GPU work.
  // Each queue maintains independent execution order.
  // Returns failure if Metal cannot allocate queue (rare, indicates resource exhaustion).
  Result<BackendQueue> create_queue(const QueueDesc& desc) override {
    @autoreleasepool {
      constexpr const char* operation = "create_queue";
      
      id<MTLCommandQueue> queue = [m_mtl_device newCommandQueue];
      if (!queue) {
        return Result<BackendQueue>::failure(
            backend_error(BackendType::Metal,
                         BackendErrorCode::RuntimeFailure,
                         operation,
                         "Failed to create Metal command queue"));
      }

      auto metal_queue = new MetalQueue(m_device, queue);
      return Result<BackendQueue>::success(
          std::shared_ptr<BackendQueue>(metal_queue),
          BackendType::Metal,
          operation);
    }
  }

  // Creates a shared event for synchronization between queues or with the CPU.
  // MTLSharedEvent supports both GPU signal/wait and CPU notify blocks.
  Result<BackendEvent> create_event(const EventDesc& desc) override {
    @autoreleasepool {
      constexpr const char* operation = "create_event";
      
      id<MTLSharedEvent> event = [m_mtl_device newSharedEvent];
      if (!event) {
        return Result<BackendEvent>::failure(
            backend_error(BackendType::Metal,
                         BackendErrorCode::RuntimeFailure,
                         operation,
                         "Failed to create Metal shared event"));
      }

      auto metal_event = new MetalEvent(m_device, event);
      return Result<BackendEvent>::success(
          std::shared_ptr<BackendEvent>(metal_event),
          BackendType::Metal,
          operation);
    }
  }

  // Allocates a GPU buffer with the specified size and alignment.
  // 
  // Implementation Details:
  // - Always uses MTLResourceStorageModeShared for unified memory access
  // - Alignment hints are informational; Metal guarantees 256-byte alignment
  // - Interop buffers (for multi-GPU) return Unsupported error
  // - Buffer contents are zero-initialized by Metal
  Result<BackendBuffer> create_buffer(const BufferDesc& desc) override {
    @autoreleasepool {
      constexpr const char* operation = "create_buffer";
      
      if (desc.size_bytes == 0) {
        return Result<BackendBuffer>::failure(
            backend_error(BackendType::Metal,
                         BackendErrorCode::InvalidArgument,
                         operation,
                         "buffer size_bytes must be > 0"));
      }
      
      if (!is_power_of_two(desc.alignment)) {
        return Result<BackendBuffer>::failure(
            backend_error(BackendType::Metal,
                         BackendErrorCode::InvalidArgument,
                         operation,
                         "buffer alignment must be a power of two"));
      }
      
      if (desc.interop_mode != BufferInteropMode::None) {
        return Result<BackendBuffer>::failure(
            backend_error(BackendType::Metal,
                         BackendErrorCode::Unsupported,
                         operation,
                         "interop buffers are not implemented for Metal runtime yet"));
      }

      // Use shared storage mode for CPU/GPU unified memory access
      // This works efficiently on both Apple Silicon (zero-copy) and Intel Macs
      MTLResourceOptions options = MTLResourceStorageModeShared;
      
      id<MTLBuffer> buffer = [m_mtl_device newBufferWithLength:desc.size_bytes
                                                       options:options];
      if (!buffer) {
        return Result<BackendBuffer>::failure(
            backend_error(BackendType::Metal,
                         BackendErrorCode::OutOfMemory,
                         operation,
                         "Metal buffer allocation failed"));
      }

      // Normalize descriptor: shared storage always provides read/write host access
      BufferDesc normalized_desc = desc;
      normalized_desc.host_access = BufferHostAccess::ReadWrite;

      auto metal_buffer = new MetalBuffer(m_device, std::move(normalized_desc), buffer);
      return Result<BackendBuffer>::success(
          std::shared_ptr<BackendBuffer>(metal_buffer),
          BackendType::Metal,
          operation);
    }
  }

  // ============================================================================
  // Synchronization Operations
  // ============================================================================

  // Records a signal operation on the event when the command buffer completes.
  // Creates a dedicated command buffer with just the signal operation.
  // 
  // Note: Each call increments the event's signal value, enabling multiple
  // signal/wait points with the same event object.
  BackendError record_event(const std::shared_ptr<BackendQueue>& queue,
                            const std::shared_ptr<BackendEvent>& event) override {
    @autoreleasepool {
      constexpr const char* operation = "record_event";
      
      auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
      if (!metal_queue) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "queue must not be null and must be a Metal queue");
      }
      
      auto metal_event = std::dynamic_pointer_cast<MetalEvent>(event);
      if (!metal_event) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "event must not be null and must be a Metal event");
      }

      id<MTLCommandBuffer> cmd_buffer = [metal_queue->queue() commandBuffer];
      if (!cmd_buffer) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::RuntimeFailure,
                            operation,
                            "Failed to create command buffer");
      }

      // Get the next signal value (atomic increment) and encode signal
      uint64_t signal_value = metal_event->get_and_increment_signal_value();
      [cmd_buffer encodeSignalEvent:metal_event->event() value:signal_value];
      [cmd_buffer commit];

      return backend_success(BackendType::Metal, operation);
    }
  }

  // Records a wait operation that blocks until the event reaches its current signal value.
  // Uses the current signal value; does not wait for future signals.
  BackendError wait_event(const std::shared_ptr<BackendQueue>& queue,
                          const std::shared_ptr<BackendEvent>& event) override {
    @autoreleasepool {
      constexpr const char* operation = "wait_event";
      
      auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
      if (!metal_queue) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "queue must not be null and must be a Metal queue");
      }
      
      auto metal_event = std::dynamic_pointer_cast<MetalEvent>(event);
      if (!metal_event) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "event must not be null and must be a Metal event");
      }

      id<MTLCommandBuffer> cmd_buffer = [metal_queue->queue() commandBuffer];
      if (!cmd_buffer) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::RuntimeFailure,
                            operation,
                            "Failed to create command buffer");
      }

      // Wait for the current signal value (non-incrementing)
      uint64_t wait_value = metal_event->current_signal_value();
      [cmd_buffer encodeWaitForEvent:metal_event->event() value:wait_value];
      [cmd_buffer commit];

      return backend_success(BackendType::Metal, operation);
    }
  }

  // Blocks the CPU until all command buffers on the queue have completed.
  // Creates an empty command buffer, commits it, and waits for completion.
  // This is a heavyweight sync; prefer event-based synchronization when possible.
  BackendError synchronize_queue(const std::shared_ptr<BackendQueue>& queue) override {
    @autoreleasepool {
      constexpr const char* operation = "synchronize_queue";
      
      auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
      if (!metal_queue) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "queue must not be null and must be a Metal queue");
      }

      id<MTLCommandBuffer> cmd_buffer = [metal_queue->queue() commandBuffer];
      if (!cmd_buffer) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::RuntimeFailure,
                            operation,
                            "Failed to create command buffer");
      }
      
      [cmd_buffer commit];
      [cmd_buffer waitUntilCompleted];

      return backend_success(BackendType::Metal, operation);
    }
  }

  // Blocks the CPU until the event is signaled.
  // Note: Current implementation is a no-op as Metal shared events don't have
  // a blocking CPU wait API. Caller should use MTLSharedEvent.notifyListener
  // or check signaledValue in a loop for CPU-side waits.
  BackendError synchronize_event(const std::shared_ptr<BackendEvent>& event) override {
    constexpr const char* operation = "synchronize_event";
    
    auto metal_event = std::dynamic_pointer_cast<MetalEvent>(event);
    if (!metal_event) {
      return backend_error(BackendType::Metal,
                          BackendErrorCode::InvalidArgument,
                          operation,
                          "event must not be null and must be a Metal event");
    }

    // TODO: Implement actual CPU wait using MTLSharedEvent.notifyListener
    // or waitUntilSignaledValue (requires spinning or callback-based approach)
    return backend_success(BackendType::Metal, operation);
  }

  // Blocks until all GPU work on the device has completed.
  // Note: Metal does not have an explicit device-wide barrier; this is a no-op.
  // For correctness, users should synchronize individual queues or events.
  BackendError synchronize_device() override {
    return backend_success(BackendType::Metal, "synchronize_device");
  }

  // ============================================================================
  // Data Transfer Operations
  // ============================================================================

  // Asynchronously copies data between two GPU buffers using a blit encoder.
  // The copy executes on the GPU; CPU continues immediately.
  // 
  // Bounds Validation:
  // - Checks both source and destination ranges against buffer sizes
  // - Detects integer overflow in offset+size calculations
  // - Zero-size copies succeed immediately
  BackendError copy_buffer_async(const std::shared_ptr<BackendQueue>& queue,
                                 const std::shared_ptr<BackendBuffer>& dst,
                                 const std::shared_ptr<BackendBuffer>& src,
                                 const CopyRegion& region) override {
    @autoreleasepool {
      constexpr const char* operation = "copy_buffer_async";
      
      auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
      if (!metal_queue) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "queue must not be null and must be a Metal queue");
      }
      
      auto metal_dst = std::dynamic_pointer_cast<MetalBuffer>(dst);
      if (!metal_dst) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "dst must not be null and must be a Metal buffer");
      }
      
      auto metal_src = std::dynamic_pointer_cast<MetalBuffer>(src);
      if (!metal_src) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "src must not be null and must be a Metal buffer");
      }

      BackendError status = validate_copy_region(metal_dst, metal_src, region, operation);
      if (!status.ok()) {
        return status;
      }
      
      if (region.size_bytes == 0) {
        return backend_success(BackendType::Metal, operation);
      }

      // Create command buffer and blit encoder for GPU copy
      id<MTLCommandBuffer> cmd_buffer = [metal_queue->queue() commandBuffer];
      if (!cmd_buffer) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::RuntimeFailure,
                            operation,
                            "Failed to create command buffer");
      }

      id<MTLBlitCommandEncoder> encoder = [cmd_buffer blitCommandEncoder];
      if (!encoder) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::RuntimeFailure,
                            operation,
                            "Failed to create blit command encoder");
      }

      [encoder copyFromBuffer:metal_src->buffer()
                 sourceOffset:region.src_offset
                     toBuffer:metal_dst->buffer()
            destinationOffset:region.dst_offset
                         size:region.size_bytes];
      
      [encoder endEncoding];
      [cmd_buffer commit];

      return backend_success(BackendType::Metal, operation);
    }
  }

  // Copies data from CPU memory to a GPU buffer.
  // 
  // Metal Unified Memory:
  // - With shared storage mode, this is a simple memcpy (no staging buffer needed)
  // - Data is immediately visible to GPU; no flush required
  // - Copy is synchronous from CPU perspective (memcpy completes before return)
  // 
  // Note: Despite the "async" naming, CPU-GPU copies in unified memory are
  // effectively synchronous since both share the same memory space.
  BackendError copy_from_host_async(const std::shared_ptr<BackendQueue>& queue,
                                    const std::shared_ptr<BackendBuffer>& dst,
                                    const void* src,
                                    const BufferTransferRegion& region) override {
    constexpr const char* operation = "copy_from_host_async";
    
    auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
    if (!metal_queue) {
      return backend_error(BackendType::Metal,
                          BackendErrorCode::InvalidArgument,
                          operation,
                          "queue must not be null and must be a Metal queue");
    }
    
    auto metal_dst = std::dynamic_pointer_cast<MetalBuffer>(dst);
    if (!metal_dst) {
      return backend_error(BackendType::Metal,
                          BackendErrorCode::InvalidArgument,
                          operation,
                          "dst must not be null and must be a Metal buffer");
    }
    
    if (region.size_bytes == 0) {
      return backend_success(BackendType::Metal, operation);
    }
    
    if (src == nullptr) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "src must not be null when size_bytes > 0");
    }
    
    if (add_overflows(region.buffer_offset, region.size_bytes) ||
        region.buffer_offset + region.size_bytes > metal_dst->size_bytes()) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "destination copy range exceeds buffer size");
    }

    // Direct memcpy into shared memory buffer
    void* dst_ptr = static_cast<char*>(metal_dst->data()) + region.buffer_offset;
    memcpy(dst_ptr, src, region.size_bytes);

    return backend_success(BackendType::Metal, operation);
  }

  // Copies data from a GPU buffer to CPU memory.
  // Similar to copy_from_host_async, uses direct memcpy for unified memory.
  BackendError copy_to_host_async(const std::shared_ptr<BackendQueue>& queue,
                                  void* dst,
                                  const std::shared_ptr<BackendBuffer>& src,
                                  const BufferTransferRegion& region) override {
    constexpr const char* operation = "copy_to_host_async";
    
    auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
    if (!metal_queue) {
      return backend_error(BackendType::Metal,
                          BackendErrorCode::InvalidArgument,
                          operation,
                          "queue must not be null and must be a Metal queue");
    }
    
    auto metal_src = std::dynamic_pointer_cast<MetalBuffer>(src);
    if (!metal_src) {
      return backend_error(BackendType::Metal,
                          BackendErrorCode::InvalidArgument,
                          operation,
                          "src must not be null and must be a Metal buffer");
    }
    
    if (region.size_bytes == 0) {
      return backend_success(BackendType::Metal, operation);
    }
    
    if (dst == nullptr) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "dst must not be null when size_bytes > 0");
    }
    
    if (add_overflows(region.buffer_offset, region.size_bytes) ||
        region.buffer_offset + region.size_bytes > metal_src->size_bytes()) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "source copy range exceeds buffer size");
    }

    // Direct memcpy from shared memory buffer
    const void* src_ptr = static_cast<const char*>(metal_src->data()) + region.buffer_offset;
    memcpy(dst, src_ptr, region.size_bytes);

    return backend_success(BackendType::Metal, operation);
  }

  // Copies between raw device pointers (both in unified memory space).
  // Used for low-level transfers outside the buffer abstraction.
  BackendError copy_device_to_host_async(const std::shared_ptr<BackendQueue>& queue,
                                         void* dst,
                                         const void* src,
                                         size_t size_bytes) override {
    constexpr const char* operation = "copy_device_to_host_async";
    
    auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
    if (!metal_queue) {
      return backend_error(BackendType::Metal,
                          BackendErrorCode::InvalidArgument,
                          operation,
                          "queue must not be null and must be a Metal queue");
    }
    
    if (size_bytes == 0) {
      return backend_success(BackendType::Metal, operation);
    }
    
    if (dst == nullptr) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "dst must not be null when size_bytes > 0");
    }
    
    if (src == nullptr) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "src must not be null when size_bytes > 0");
    }

    memcpy(dst, src, size_bytes);

    return backend_success(BackendType::Metal, operation);
  }

  // Copies between raw device pointers (both in unified memory space).
  // Identical to copy_device_to_host_async; both are simple memcpys.
  BackendError copy_host_to_device_async(const std::shared_ptr<BackendQueue>& queue,
                                          void* dst,
                                          const void* src,
                                          size_t size_bytes) override {
    constexpr const char* operation = "copy_host_to_device_async";
    
    auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
    if (!metal_queue) {
      return backend_error(BackendType::Metal,
                          BackendErrorCode::InvalidArgument,
                          operation,
                          "queue must not be null and must be a Metal queue");
    }
    
    if (size_bytes == 0) {
      return backend_success(BackendType::Metal, operation);
    }
    
    if (dst == nullptr) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "dst must not be null when size_bytes > 0");
    }
    
    if (src == nullptr) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "src must not be null when size_bytes > 0");
    }

    memcpy(dst, src, size_bytes);

    return backend_success(BackendType::Metal, operation);
  }

  // Fills a region of a buffer with a constant byte value using GPU blit operation.
  // Useful for zeroing buffers or initializing with a pattern.
  // Executes asynchronously on the GPU via a blit command encoder.
  BackendError fill_buffer_async(const std::shared_ptr<BackendQueue>& queue,
                                 const std::shared_ptr<BackendBuffer>& buffer,
                                 uint8_t value,
                                 size_t offset,
                                 size_t size_bytes) override {
    @autoreleasepool {
      constexpr const char* operation = "fill_buffer_async";
      
      auto metal_queue = std::dynamic_pointer_cast<MetalQueue>(queue);
      if (!metal_queue) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "queue must not be null and must be a Metal queue");
      }
      
      auto metal_buffer = std::dynamic_pointer_cast<MetalBuffer>(buffer);
      if (!metal_buffer) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::InvalidArgument,
                            operation,
                            "buffer must not be null and must be a Metal buffer");
      }
      
      if (size_bytes == 0) {
        return backend_success(BackendType::Metal, operation);
      }
      
      if (add_overflows(offset, size_bytes) ||
          offset + size_bytes > metal_buffer->size_bytes()) {
        return backend_error(BackendType::Metal,
                             BackendErrorCode::InvalidArgument,
                             operation,
                             "fill range exceeds buffer size");
      }

      // Create blit encoder for GPU fill operation
      id<MTLCommandBuffer> cmd_buffer = [metal_queue->queue() commandBuffer];
      if (!cmd_buffer) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::RuntimeFailure,
                            operation,
                            "Failed to create command buffer");
      }

      id<MTLBlitCommandEncoder> encoder = [cmd_buffer blitCommandEncoder];
      if (!encoder) {
        return backend_error(BackendType::Metal,
                            BackendErrorCode::RuntimeFailure,
                            operation,
                            "Failed to create blit command encoder");
      }

      NSRange range = NSMakeRange(offset, size_bytes);
      [encoder fillBuffer:metal_buffer->buffer() range:range value:value];
      
      [encoder endEncoding];
      [cmd_buffer commit];

      return backend_success(BackendType::Metal, operation);
    }
  }

private:
  // Validates buffer copy parameters, checking for null buffers and out-of-bounds access.
  // Returns an error if any validation fails, otherwise returns success.
  BackendError validate_copy_region(const std::shared_ptr<MetalBuffer>& dst,
                                    const std::shared_ptr<MetalBuffer>& src,
                                    const CopyRegion& region,
                                    const std::string& operation) const {
    if (!dst || !src) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "copy buffers must not be null");
    }
    
    // Check destination bounds with overflow protection
    if (add_overflows(region.dst_offset, region.size_bytes) ||
        region.dst_offset + region.size_bytes > dst->size_bytes()) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "destination copy range exceeds buffer size");
    }
    
    // Check source bounds with overflow protection
    if (add_overflows(region.src_offset, region.size_bytes) ||
        region.src_offset + region.size_bytes > src->size_bytes()) {
      return backend_error(BackendType::Metal,
                           BackendErrorCode::InvalidArgument,
                           operation,
                           "source copy range exceeds buffer size");
    }
    
    return backend_success(BackendType::Metal, operation);
  }

private:
  int m_device = 0;
  id<MTLDevice> m_mtl_device = nullptr;
  CapabilityProfile m_capability_profile;
};

}  // namespace

// ============================================================================
// Factory Function
// ============================================================================

// Creates a Metal backend runtime instance for the specified device index.
// 
// Parameters:
//   device: Zero-based GPU device index (0 = default/integrated GPU)
// 
// Returns:
//   Result containing either a valid BackendRuntime or an error
// 
// Thread Safety:
//   Safe to call from any thread; creates independent runtime instance
// 
// Exception Handling:
//   Catches Objective-C exceptions and converts to BackendError
// 
// Usage:
//   auto result = create_metal_backend_runtime(0);
//   if (result.ok()) {
//     auto runtime = result.value();
//     // Use runtime for GPU operations
//   }
Result<BackendRuntime> create_metal_backend_runtime(int device) {
  constexpr const char* operation = "create_metal_backend_runtime";
  
  @autoreleasepool {
    @try {
      auto runtime = new MetalRuntime(device);
      return Result<BackendRuntime>::success(
          std::shared_ptr<BackendRuntime>(runtime),
          BackendType::Metal,
          operation);
    } @catch (NSException* exception) {
      // Convert Objective-C exception to BackendError
      NSString* reason = exception.reason ?: @"Unknown Metal initialization error";
      return Result<BackendRuntime>::failure(
          backend_error(BackendType::Metal,
                       BackendErrorCode::RuntimeFailure,
                       operation,
                       [reason UTF8String]));
    }
  }
}

}  // namespace tinygs
