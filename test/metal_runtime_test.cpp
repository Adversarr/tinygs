#include <gtest/gtest.h>
#include "tinygs/platform/runtime_factory.hpp"
#include <limits>
#include <vector>
#include <cstring>

namespace tinygs {

class MetalRuntimeTest : public ::testing::Test {
protected:
  void SetUp() override {
    BackendConfig config{BackendType::Metal, 0};
    auto result = create_backend_runtime(config);
    ASSERT_TRUE(result.ok()) << result.error().message;
    runtime = result.value();
  }

  std::shared_ptr<BackendRuntime> runtime;
};

TEST_F(MetalRuntimeTest, DeviceCreation) {
  EXPECT_EQ(runtime->backend_type(), BackendType::Metal);
  EXPECT_GE(runtime->device(), 0);

  auto caps = runtime->capability_profile();
  EXPECT_TRUE(caps.supports_queues);
  EXPECT_TRUE(caps.supports_events);
  EXPECT_TRUE(caps.supports_device_buffers);
  EXPECT_TRUE(caps.supports_unified_memory);
}

TEST_F(MetalRuntimeTest, QueueCreation) {
  QueueDesc desc;
  auto result = runtime->create_queue(desc);
  EXPECT_TRUE(result.ok()) << result.error().message;
  EXPECT_NE(result.value(), nullptr);
  EXPECT_EQ(result.value()->backend_type(), BackendType::Metal);
  EXPECT_NE(result.value()->native_handle(), nullptr);
}

TEST_F(MetalRuntimeTest, EventCreation) {
  EventDesc desc;
  auto result = runtime->create_event(desc);
  EXPECT_TRUE(result.ok()) << result.error().message;
  EXPECT_NE(result.value(), nullptr);
  EXPECT_EQ(result.value()->backend_type(), BackendType::Metal);
  EXPECT_NE(result.value()->native_handle(), nullptr);
}

TEST_F(MetalRuntimeTest, BufferAllocation) {
  BufferDesc desc;
  desc.size_bytes = 1024;
  desc.memory_class = BufferMemoryClass::Device;

  auto result = runtime->create_buffer(desc);
  EXPECT_TRUE(result.ok()) << result.error().message;
  EXPECT_NE(result.value(), nullptr);
  EXPECT_EQ(result.value()->size_bytes(), 1024);
  EXPECT_EQ(result.value()->backend_type(), BackendType::Metal);
}

TEST_F(MetalRuntimeTest, BufferZeroSize) {
  BufferDesc desc;
  desc.size_bytes = 0;

  auto result = runtime->create_buffer(desc);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, BackendErrorCode::InvalidArgument);
}

TEST_F(MetalRuntimeTest, BufferInvalidAlignment) {
  BufferDesc desc;
  desc.size_bytes = 1024;
  desc.alignment = 100;  // Not power of two

  auto result = runtime->create_buffer(desc);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, BackendErrorCode::InvalidArgument);
}

TEST_F(MetalRuntimeTest, UnifiedMemoryAccess) {
  BufferDesc desc;
  desc.size_bytes = 256;
  desc.memory_class = BufferMemoryClass::Unified;

  auto result = runtime->create_buffer(desc);
  ASSERT_TRUE(result.ok()) << result.error().message;
  auto buffer = result.value();

  EXPECT_NE(buffer->data(), nullptr);
}

TEST_F(MetalRuntimeTest, HostToDeviceRoundtrip) {
  BufferDesc desc;
  desc.size_bytes = 256;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::vector<uint8_t> src(256);
  for (size_t i = 0; i < 256; ++i) {
    src[i] = static_cast<uint8_t>(i);
  }

  auto copyResult = runtime->copy_from_host_async(
      *queue, *buffer, src.data(), BufferTransferRegion{256, 0});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  std::vector<uint8_t> dst(256);
  copyResult = runtime->copy_to_host_async(
      *queue, dst.data(), *buffer, BufferTransferRegion{256, 0});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  EXPECT_EQ(src, dst);
}

TEST_F(MetalRuntimeTest, BufferFill) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  auto fillResult = runtime->fill_buffer_async(*queue, *buffer, 0x42);
  EXPECT_TRUE(fillResult.ok()) << fillResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  uint8_t* data = static_cast<uint8_t*>(buffer->data());
  for (size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(data[i], 0x42) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, BufferFillWithOffset) {
  BufferDesc desc;
  desc.size_bytes = 256;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  auto fillResult = runtime->fill_buffer_async(*queue, *buffer, 0, 0, 256);
  EXPECT_TRUE(fillResult.ok()) << fillResult.message;

  fillResult = runtime->fill_buffer_async(*queue, *buffer, 0xAB, 100, 50);
  EXPECT_TRUE(fillResult.ok()) << fillResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  uint8_t* data = static_cast<uint8_t*>(buffer->data());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_EQ(data[i], 0) << "Mismatch at index " << i;
  }
  for (size_t i = 100; i < 150; ++i) {
    EXPECT_EQ(data[i], 0xAB) << "Mismatch at index " << i;
  }
  for (size_t i = 150; i < 256; ++i) {
    EXPECT_EQ(data[i], 0) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, EventSynchronization) {
  auto eventResult = runtime->create_event(EventDesc{});
  ASSERT_TRUE(eventResult.ok()) << eventResult.error().message;
  auto event = eventResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  auto recordResult = runtime->record_event(*queue, *event);
  EXPECT_TRUE(recordResult.ok()) << recordResult.message;

  auto waitResult = runtime->wait_event(*queue, *event);
  EXPECT_TRUE(waitResult.ok()) << waitResult.message;

  auto syncResult = runtime->synchronize_event(*event);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;
}

TEST_F(MetalRuntimeTest, CopyBufferAsync) {
  BufferDesc desc;
  desc.size_bytes = 256;

  auto srcBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(srcBufferResult.ok()) << srcBufferResult.error().message;
  auto srcBuffer = srcBufferResult.value();

  auto dstBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(dstBufferResult.ok()) << dstBufferResult.error().message;
  auto dstBuffer = dstBufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::vector<uint8_t> srcData(256);
  for (size_t i = 0; i < 256; ++i) {
    srcData[i] = static_cast<uint8_t>(i * 2);
  }

  auto copyResult = runtime->copy_from_host_async(
      *queue, *srcBuffer, srcData.data(), BufferTransferRegion{256, 0});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  copyResult = runtime->copy_buffer_async(
      *queue, *dstBuffer, *srcBuffer, CopyRegion{256, 0, 0});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  uint8_t* dstData = static_cast<uint8_t*>(dstBuffer->data());
  for (size_t i = 0; i < 256; ++i) {
    EXPECT_EQ(dstData[i], srcData[i]) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, CopyBufferWithOffsets) {
  BufferDesc desc;
  desc.size_bytes = 512;

  auto srcBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(srcBufferResult.ok()) << srcBufferResult.error().message;
  auto srcBuffer = srcBufferResult.value();

  auto dstBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(dstBufferResult.ok()) << dstBufferResult.error().message;
  auto dstBuffer = dstBufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::vector<uint8_t> srcData(512, 0xCD);
  auto copyResult = runtime->copy_from_host_async(
      *queue, *srcBuffer, srcData.data(), BufferTransferRegion{512, 0});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  copyResult = runtime->copy_buffer_async(
      *queue, *dstBuffer, *srcBuffer, CopyRegion{128, 64, 128});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  uint8_t* dstData = static_cast<uint8_t*>(dstBuffer->data());
  for (size_t i = 64; i < 64 + 128; ++i) {
    EXPECT_EQ(dstData[i], 0xCD) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, CopyDeviceToHostAsync) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  uint8_t* devicePtr = static_cast<uint8_t*>(buffer->data());
  for (size_t i = 0; i < 128; ++i) {
    devicePtr[i] = static_cast<uint8_t>(i + 10);
  }

  std::vector<uint8_t> hostData(128);
  auto copyResult = runtime->copy_device_to_host_async(
      *queue, hostData.data(), devicePtr, 128);
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  for (size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(hostData[i], i + 10) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, CopyHostToDeviceAsync) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::vector<uint8_t> hostData(128);
  for (size_t i = 0; i < 128; ++i) {
    hostData[i] = static_cast<uint8_t>(255 - i);
  }

  uint8_t* devicePtr = static_cast<uint8_t*>(buffer->data());
  auto copyResult = runtime->copy_host_to_device_async(
      *queue, devicePtr, hostData.data(), 128);
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  for (size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(devicePtr[i], 255 - i) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, MultipleQueues) {
  auto queue1Result = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queue1Result.ok());
  auto queue1 = queue1Result.value();

  auto queue2Result = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queue2Result.ok());
  auto queue2 = queue2Result.value();

  BufferDesc desc;
  desc.size_bytes = 64;

  auto buffer1Result = runtime->create_buffer(desc);
  auto buffer1 = buffer1Result.value();

  auto buffer2Result = runtime->create_buffer(desc);
  auto buffer2 = buffer2Result.value();

  auto fill1 = runtime->fill_buffer_async(*queue1, *buffer1, 0x11);
  auto fill2 = runtime->fill_buffer_async(*queue2, *buffer2, 0x22);

  EXPECT_TRUE(fill1.ok());
  EXPECT_TRUE(fill2.ok());

  runtime->synchronize_queue(*queue1);
  runtime->synchronize_queue(*queue2);

  uint8_t* data1 = static_cast<uint8_t*>(buffer1->data());
  uint8_t* data2 = static_cast<uint8_t*>(buffer2->data());

  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(data1[i], 0x11);
    EXPECT_EQ(data2[i], 0x22);
  }
}

TEST_F(MetalRuntimeTest, SynchronizeDevice) {
  auto result = runtime->synchronize_device();
  EXPECT_TRUE(result.ok());
}

TEST_F(MetalRuntimeTest, InteropBufferUnsupported) {
  BufferDesc desc;
  desc.size_bytes = 1024;
  desc.interop_mode = BufferInteropMode::External;

  auto result = runtime->create_buffer(desc);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, BackendErrorCode::Unsupported);
}

TEST_F(MetalRuntimeTest, CapabilityProfile) {
  auto caps = runtime->capability_profile();
  
  EXPECT_TRUE(caps.supports_queues);
  EXPECT_TRUE(caps.supports_events);
  EXPECT_TRUE(caps.supports_device_buffers);
  EXPECT_TRUE(caps.supports_unified_memory);
  EXPECT_TRUE(caps.supports_host_visible_buffers);
  EXPECT_FALSE(caps.supports_interop_buffers);
  EXPECT_FALSE(caps.supports_graph_capture);
}

// ============================================================================
// NVI Validation Edge Cases
// ============================================================================

TEST_F(MetalRuntimeTest, CopyRegionOverflow) {
  BufferDesc desc;
  desc.size_bytes = 256;

  auto srcBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(srcBufferResult.ok()) << srcBufferResult.error().message;
  auto srcBuffer = srcBufferResult.value();

  auto dstBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(dstBufferResult.ok()) << dstBufferResult.error().message;
  auto dstBuffer = dstBufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  CopyRegion region;
  region.size_bytes = 16;
  region.dst_offset = std::numeric_limits<size_t>::max() - 8;
  region.src_offset = 0;

  auto result = runtime->copy_buffer_async(*queue, *dstBuffer, *srcBuffer, region);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);
}

TEST_F(MetalRuntimeTest, CopyRegionExceedsBufferSize) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto srcBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(srcBufferResult.ok()) << srcBufferResult.error().message;
  auto srcBuffer = srcBufferResult.value();

  auto dstBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(dstBufferResult.ok()) << dstBufferResult.error().message;
  auto dstBuffer = dstBufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  CopyRegion region;
  region.size_bytes = 64;
  region.dst_offset = 100;
  region.src_offset = 0;

  auto result = runtime->copy_buffer_async(*queue, *dstBuffer, *srcBuffer, region);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);

  region.dst_offset = 0;
  region.src_offset = 100;
  result = runtime->copy_buffer_async(*queue, *dstBuffer, *srcBuffer, region);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);
}

TEST_F(MetalRuntimeTest, BufferTransferRegionOverflow) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::vector<uint8_t> hostData(128);

  BufferTransferRegion region;
  region.size_bytes = 64;
  region.buffer_offset = std::numeric_limits<size_t>::max() - 32;

  auto result = runtime->copy_from_host_async(*queue, *buffer, hostData.data(), region);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);

  result = runtime->copy_to_host_async(*queue, hostData.data(), *buffer, region);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);
}

TEST_F(MetalRuntimeTest, NullHostPointer) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  BufferTransferRegion region;
  region.size_bytes = 64;
  region.buffer_offset = 0;

  auto result = runtime->copy_from_host_async(*queue, *buffer, nullptr, region);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);

  result = runtime->copy_to_host_async(*queue, nullptr, *buffer, region);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);
}

TEST_F(MetalRuntimeTest, ZeroSizeOperationsReturnSuccess) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto srcBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(srcBufferResult.ok()) << srcBufferResult.error().message;
  auto srcBuffer = srcBufferResult.value();

  auto dstBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(dstBufferResult.ok()) << dstBufferResult.error().message;
  auto dstBuffer = dstBufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  auto result = runtime->fill_buffer_async(*queue, *dstBuffer, 0x42, 0, 0);
  EXPECT_TRUE(result.ok());

  auto copyResult = runtime->copy_buffer_async(
      *queue, *dstBuffer, *srcBuffer, CopyRegion{0, 0, 0});
  EXPECT_TRUE(copyResult.ok());

  std::vector<uint8_t> hostData(1);
  auto hostCopyResult = runtime->copy_from_host_async(
      *queue, *dstBuffer, hostData.data(), BufferTransferRegion{0, 0});
  EXPECT_TRUE(hostCopyResult.ok());

  hostCopyResult = runtime->copy_to_host_async(
      *queue, hostData.data(), *dstBuffer, BufferTransferRegion{0, 0});
  EXPECT_TRUE(hostCopyResult.ok());

  auto rawResult = runtime->copy_device_to_host_async(*queue, nullptr, nullptr, 0);
  EXPECT_TRUE(rawResult.ok());

  rawResult = runtime->copy_host_to_device_async(*queue, nullptr, nullptr, 0);
  EXPECT_TRUE(rawResult.ok());
}

TEST_F(MetalRuntimeTest, FillRangeOverflow) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  auto result = runtime->fill_buffer_async(*queue, *buffer, 0x42, 100, 50);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);

  result = runtime->fill_buffer_async(*queue, *buffer, 0x42, 0, 200);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code, BackendErrorCode::InvalidArgument);
}

TEST_F(MetalRuntimeTest, DeviceBufferWithHostAccess) {
  BufferDesc desc;
  desc.size_bytes = 128;
  desc.memory_class = BufferMemoryClass::Device;
  desc.host_access = BufferHostAccess::ReadWrite;

  auto result = runtime->create_buffer(desc);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, BackendErrorCode::Unsupported);
}

// ============================================================================
// Event Synchronization Edge Cases
// ============================================================================

TEST_F(MetalRuntimeTest, EventReuseMultipleCycles) {
  auto eventResult = runtime->create_event(EventDesc{});
  ASSERT_TRUE(eventResult.ok()) << eventResult.error().message;
  auto event = eventResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  for (int i = 0; i < 5; ++i) {
    auto recordResult = runtime->record_event(*queue, *event);
    EXPECT_TRUE(recordResult.ok()) << recordResult.message;

    auto syncResult = runtime->synchronize_event(*event);
    EXPECT_TRUE(syncResult.ok()) << syncResult.message;

    auto waitResult = runtime->wait_event(*queue, *event);
    EXPECT_TRUE(waitResult.ok()) << waitResult.message;
  }
}

TEST_F(MetalRuntimeTest, CrossQueueEventSync) {
  auto queue1Result = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queue1Result.ok()) << queue1Result.error().message;
  auto queue1 = queue1Result.value();

  auto queue2Result = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queue2Result.ok()) << queue2Result.error().message;
  auto queue2 = queue2Result.value();

  auto eventResult = runtime->create_event(EventDesc{});
  ASSERT_TRUE(eventResult.ok()) << eventResult.error().message;
  auto event = eventResult.value();

  BufferDesc desc;
  desc.size_bytes = 128;
  auto srcBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(srcBufferResult.ok()) << srcBufferResult.error().message;
  auto srcBuffer = srcBufferResult.value();
  auto dstBufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(dstBufferResult.ok()) << dstBufferResult.error().message;
  auto dstBuffer = dstBufferResult.value();

  std::vector<uint8_t> srcData(128);
  for (int i = 0; i < 128; ++i) {
    srcData[i] = static_cast<uint8_t>(i + 17);
  }

  auto copyResult = runtime->copy_from_host_async(
      *queue1, *srcBuffer, srcData.data(), BufferTransferRegion{128, 0});
  ASSERT_TRUE(copyResult.ok()) << copyResult.message;

  auto recordResult = runtime->record_event(*queue1, *event);
  ASSERT_TRUE(recordResult.ok()) << recordResult.message;

  auto waitResult = runtime->wait_event(*queue2, *event);
  ASSERT_TRUE(waitResult.ok()) << waitResult.message;

  copyResult = runtime->copy_buffer_async(
      *queue2, *dstBuffer, *srcBuffer, CopyRegion{128, 0, 0});
  ASSERT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue2);
  ASSERT_TRUE(syncResult.ok()) << syncResult.message;

  uint8_t* dstData = static_cast<uint8_t*>(dstBuffer->data());
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(dstData[i], srcData[i]) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, SynchronizeEventAlreadySignaled) {
  auto eventResult = runtime->create_event(EventDesc{});
  ASSERT_TRUE(eventResult.ok()) << eventResult.error().message;
  auto event = eventResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  auto recordResult = runtime->record_event(*queue, *event);
  ASSERT_TRUE(recordResult.ok()) << recordResult.message;

  auto syncResult = runtime->synchronize_event(*event);
  ASSERT_TRUE(syncResult.ok()) << syncResult.message;

  syncResult = runtime->synchronize_event(*event);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;
}

// ============================================================================
// Buffer Edge Cases
// ============================================================================

TEST_F(MetalRuntimeTest, LargeBufferAllocation) {
  BufferDesc desc;
  desc.size_bytes = 100 * 1024 * 1024;

  auto result = runtime->create_buffer(desc);
  if (result.ok()) {
    EXPECT_NE(result.value(), nullptr);
    EXPECT_EQ(result.value()->size_bytes(), desc.size_bytes);
  } else {
    EXPECT_EQ(result.error().code, BackendErrorCode::OutOfMemory);
  }
}

TEST_F(MetalRuntimeTest, BufferMemoryClassUnified) {
  BufferDesc desc;
  desc.size_bytes = 256;
  desc.memory_class = BufferMemoryClass::Unified;

  auto result = runtime->create_buffer(desc);
  ASSERT_TRUE(result.ok()) << result.error().message;
  auto buffer = result.value();

  EXPECT_NE(buffer->data(), nullptr);
  EXPECT_EQ(buffer->size_bytes(), 256);
}

TEST_F(MetalRuntimeTest, BufferMemoryClassHostPinned) {
  BufferDesc desc;
  desc.size_bytes = 256;
  desc.memory_class = BufferMemoryClass::HostPinned;

  auto result = runtime->create_buffer(desc);
  ASSERT_TRUE(result.ok()) << result.error().message;
  auto buffer = result.value();

  EXPECT_NE(buffer->data(), nullptr);
  EXPECT_EQ(buffer->size_bytes(), 256);

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::vector<uint8_t> srcData(256);
  for (size_t i = 0; i < 256; ++i) {
    srcData[i] = static_cast<uint8_t>(i * 3);
  }

  auto copyResult = runtime->copy_from_host_async(
      *queue, *buffer, srcData.data(), BufferTransferRegion{256, 0});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  std::vector<uint8_t> dstData(256);
  copyResult = runtime->copy_to_host_async(
      *queue, dstData.data(), *buffer, BufferTransferRegion{256, 0});
  EXPECT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  EXPECT_TRUE(syncResult.ok()) << syncResult.message;

  EXPECT_EQ(srcData, dstData);
}

TEST_F(MetalRuntimeTest, BufferAlignmentVariations) {
  std::vector<size_t> alignments = {1, 2, 4, 256, 4096};

  for (size_t alignment : alignments) {
    BufferDesc desc;
    desc.size_bytes = 1024;
    desc.alignment = alignment;

    auto result = runtime->create_buffer(desc);
    EXPECT_TRUE(result.ok()) << "Alignment " << alignment << " failed: " << result.error().message;
    if (result.ok()) {
      EXPECT_NE(result.value(), nullptr);
    }
  }
}

// ============================================================================
// Metal-Specific Behavior
// ============================================================================

TEST_F(MetalRuntimeTest, UnifiedMemoryDirectAccess) {
  BufferDesc desc;
  desc.size_bytes = 128;

  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  uint8_t* data = static_cast<uint8_t*>(buffer->data());
  ASSERT_NE(data, nullptr);

  for (size_t i = 0; i < 128; ++i) {
    data[i] = static_cast<uint8_t>(i + 42);
  }

  for (size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(data[i], static_cast<uint8_t>(i + 42)) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, CapabilityProfileMetalSpecific) {
  auto caps = runtime->capability_profile();

  EXPECT_TRUE(caps.supports_unified_memory);
  EXPECT_TRUE(caps.supports_host_visible_buffers);
  EXPECT_FALSE(caps.supports_interop_buffers);
  EXPECT_FALSE(caps.supports_graph_capture);
}

TEST_F(MetalRuntimeTest, CopyToHostRespectsQueuedWrites) {
  BufferDesc desc;
  desc.size_bytes = 1 << 20;
  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::memset(buffer->data(), 0x00, desc.size_bytes);
  auto fillResult = runtime->fill_buffer_async(*queue, *buffer, 0x7B);
  ASSERT_TRUE(fillResult.ok()) << fillResult.message;

  std::vector<uint8_t> host(desc.size_bytes, 0x00);
  auto copyResult = runtime->copy_to_host_async(
      *queue, host.data(), *buffer, BufferTransferRegion{desc.size_bytes, 0});
  ASSERT_TRUE(copyResult.ok()) << copyResult.message;

  for (size_t i = 0; i < host.size(); ++i) {
    EXPECT_EQ(host[i], 0x7B) << "Mismatch at index " << i;
  }
}

TEST_F(MetalRuntimeTest, CopyFromHostRespectsQueuedWrites) {
  BufferDesc desc;
  desc.size_bytes = 1 << 20;
  auto bufferResult = runtime->create_buffer(desc);
  ASSERT_TRUE(bufferResult.ok()) << bufferResult.error().message;
  auto buffer = bufferResult.value();

  auto queueResult = runtime->create_queue(QueueDesc{});
  ASSERT_TRUE(queueResult.ok()) << queueResult.error().message;
  auto queue = queueResult.value();

  std::memset(buffer->data(), 0x00, desc.size_bytes);
  auto fillResult = runtime->fill_buffer_async(*queue, *buffer, 0xAA);
  ASSERT_TRUE(fillResult.ok()) << fillResult.message;

  std::vector<uint8_t> host(desc.size_bytes, 0x11);
  auto copyResult = runtime->copy_from_host_async(
      *queue, *buffer, host.data(), BufferTransferRegion{desc.size_bytes, 0});
  ASSERT_TRUE(copyResult.ok()) << copyResult.message;

  auto syncResult = runtime->synchronize_queue(*queue);
  ASSERT_TRUE(syncResult.ok()) << syncResult.message;

  uint8_t* data = static_cast<uint8_t*>(buffer->data());
  for (size_t i = 0; i < desc.size_bytes; ++i) {
    EXPECT_EQ(data[i], 0x11) << "Mismatch at index " << i;
  }
}

}  // namespace tinygs
