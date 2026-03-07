#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "tinygs/platform/buffer_utils.hpp"

namespace {

using namespace tinygs;

class FakeQueue final : public BackendQueue {
public:
  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return 0; }
  void* native_handle() const noexcept override { return nullptr; }
};

class FakeEvent final : public BackendEvent {
public:
  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return 0; }
  void* native_handle() const noexcept override { return nullptr; }
};

class FakeBuffer final : public BackendBuffer {
public:
  explicit FakeBuffer(BufferDesc desc) : m_desc(std::move(desc)), m_storage(m_desc.size_bytes, 0) {}

  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return 0; }
  size_t size_bytes() const noexcept override { return m_storage.size(); }
  const BufferDesc& desc() const noexcept override { return m_desc; }
  void* data() const noexcept override { return const_cast<uint8_t*>(m_storage.data()); }
  void* native_handle() const noexcept override { return data(); }

private:
  BufferDesc m_desc;
  std::vector<uint8_t> m_storage;
};

class FakeRuntime final : public BackendRuntime {
public:
  int sync_queue_calls = 0;

  BackendType backend_type() const noexcept override { return BackendType::Cuda; }
  int device() const noexcept override { return 0; }
  CapabilityProfile capability_profile() const override { return CapabilityProfile{}; }

protected:
  Result<BackendQueue> do_create_queue(const QueueDesc&) override {
    return Result<BackendQueue>::success(std::make_shared<FakeQueue>(), backend_type(), "create_queue");
  }
  Result<BackendEvent> do_create_event(const EventDesc&) override {
    return Result<BackendEvent>::success(std::make_shared<FakeEvent>(), backend_type(), "create_event");
  }
  Result<BackendBuffer> do_create_buffer(const BufferDesc& desc) override {
    return Result<BackendBuffer>::success(
        std::make_shared<FakeBuffer>(desc), backend_type(), "create_buffer");
  }

  BackendError do_record_event(BackendQueue&, BackendEvent&) override {
    return backend_success(backend_type(), "record_event");
  }
  BackendError do_wait_event(BackendQueue&, BackendEvent&) override {
    return backend_success(backend_type(), "wait_event");
  }
  BackendError do_synchronize_queue(BackendQueue&) override {
    sync_queue_calls += 1;
    return backend_success(backend_type(), "synchronize_queue");
  }
  BackendError do_synchronize_event(BackendEvent&) override {
    return backend_success(backend_type(), "synchronize_event");
  }
  BackendError do_synchronize_device() override {
    return backend_success(backend_type(), "synchronize_device");
  }

  BackendError do_copy_buffer(
      BackendQueue&,
      BackendBuffer& dst, size_t dst_offset,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    std::memmove(
        static_cast<uint8_t*>(dst.data()) + dst_offset,
        static_cast<const uint8_t*>(src.data()) + src_offset,
        size_bytes);
    return backend_success(backend_type(), "copy_buffer_async");
  }
  BackendError do_copy_from_host(
      BackendQueue&,
      BackendBuffer& dst, size_t dst_offset,
      const void* src, size_t size_bytes) override {
    std::memmove(
        static_cast<uint8_t*>(dst.data()) + dst_offset,
        src,
        size_bytes);
    return backend_success(backend_type(), "copy_from_host_async");
  }
  BackendError do_copy_to_host(
      BackendQueue&,
      void* dst,
      BackendBuffer& src, size_t src_offset,
      size_t size_bytes) override {
    std::memmove(
        dst,
        static_cast<const uint8_t*>(src.data()) + src_offset,
        size_bytes);
    return backend_success(backend_type(), "copy_to_host_async");
  }
  BackendError do_transfer_raw(
      BackendQueue&,
      void* dst, const void* src,
      size_t size_bytes,
      TransferDirection) override {
    std::memmove(dst, src, size_bytes);
    return backend_success(backend_type(), "transfer_raw");
  }
  BackendError do_fill_buffer(
      BackendQueue&,
      BackendBuffer& buffer,
      size_t offset, uint8_t value, size_t size_bytes) override {
    std::memset(static_cast<uint8_t*>(buffer.data()) + offset, static_cast<int>(value), size_bytes);
    return backend_success(backend_type(), "fill_buffer_async");
  }
};

std::shared_ptr<BackendQueue> create_test_queue(const std::shared_ptr<FakeRuntime>& runtime) {
  auto result = runtime->create_queue({});
  EXPECT_TRUE(result.ok());
  return result.value();
}

TEST(BufferUtilsTest, FillBufferAsyncDoesNotSynchronizeAndUsesByteValue) {
  auto runtime = std::make_shared<FakeRuntime>();
  auto queue = create_test_queue(runtime);
  auto buffer = create_device_buffer_for<uint32_t>(*runtime, 4, "fill_async_buffer");

  fill_buffer_async(*runtime, *queue, buffer, 0xAB);

  EXPECT_EQ(runtime->sync_queue_calls, 0);

  std::vector<uint8_t> host(buffer->size_bytes(), 0);
  copy_to_host(*runtime, *queue, buffer, host.data(), host.size());
  EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](uint8_t value) { return value == 0xAB; }));
}

TEST(BufferUtilsTest, FillBufferSynchronizesExactlyOnce) {
  auto runtime = std::make_shared<FakeRuntime>();
  auto queue = create_test_queue(runtime);
  auto buffer = create_device_buffer_for<uint32_t>(*runtime, 2, "fill_sync_buffer");

  fill_buffer(*runtime, *queue, buffer, 0x11);

  EXPECT_EQ(runtime->sync_queue_calls, 1);
}

TEST(BufferUtilsTest, ResizeBufferAsyncKeepsPrefixWithoutSynchronizing) {
  auto runtime = std::make_shared<FakeRuntime>();
  auto queue = create_test_queue(runtime);
  auto old_buffer = create_device_buffer_for<uint32_t>(*runtime, 3, "old_buffer");
  std::vector<uint32_t> initial{7, 8, 9};
  copy_from_host(*runtime, *queue, old_buffer, initial);
  const int baseline_syncs = runtime->sync_queue_calls;

  auto resized = resize_buffer_async<uint32_t>(*runtime, *queue, old_buffer, 5, "resized_buffer");

  EXPECT_EQ(runtime->sync_queue_calls, baseline_syncs);

  std::vector<uint32_t> host;
  copy_to_host(*runtime, *queue, resized, host);
  ASSERT_EQ(host.size(), 5u);
  EXPECT_EQ(host[0], 7u);
  EXPECT_EQ(host[1], 8u);
  EXPECT_EQ(host[2], 9u);
  EXPECT_EQ(host[3], 0u);
  EXPECT_EQ(host[4], 0u);
}

TEST(BufferUtilsTest, ResizeBufferSynchronizesForFillAndCopy) {
  auto runtime = std::make_shared<FakeRuntime>();
  auto queue = create_test_queue(runtime);
  auto old_buffer = create_device_buffer_for<uint32_t>(*runtime, 2, "old_buffer");
  std::vector<uint32_t> initial{3, 4};
  copy_from_host(*runtime, *queue, old_buffer, initial);
  const int baseline_syncs = runtime->sync_queue_calls;

  auto resized = resize_buffer<uint32_t>(*runtime, *queue, old_buffer, 4, "resized_buffer");

  ASSERT_NE(resized, nullptr);
  EXPECT_EQ(runtime->sync_queue_calls, baseline_syncs + 2);
}

TEST(BufferUtilsTest, RawCopyHelpersRoundTripData) {
  auto runtime = std::make_shared<FakeRuntime>();
  auto queue = create_test_queue(runtime);
  std::vector<uint32_t> src{1, 2, 3, 4};
  std::vector<uint32_t> device(src.size(), 0);
  std::vector<uint32_t> dst(src.size(), 0);

  copy_raw_host_to_device_async(runtime, queue, device.data(), src.data(), src.size() * sizeof(uint32_t));
  EXPECT_EQ(runtime->sync_queue_calls, 0);

  copy_raw_device_to_host(runtime, queue, dst.data(), device.data(), dst.size() * sizeof(uint32_t));

  EXPECT_EQ(dst, src);
  EXPECT_EQ(runtime->sync_queue_calls, 1);
}

}  // namespace
