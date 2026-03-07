#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tinygs/optim/optim.hpp>
#include <tinygs/platform/backend_build.hpp>
#include <tinygs/platform/runtime_factory.hpp>
#include <tinygs/cuda/common_host.hpp>

namespace {
using namespace tinygs;

struct OptimizerFactoryTest : public ::testing::Test {
  void SetUp() override {
    try {
      if (cuda_device_count() <= 0) { GTEST_SKIP() << "No CUDA device"; return; }
    } catch (...) { GTEST_SKIP() << "No CUDA device"; return; }
    BackendConfig config;
    config.type = compiled_backend_type();
    config.device = 0;
    auto r = create_backend_runtime(config);
    if (!r.ok()) { GTEST_SKIP() << "Failed to create runtime"; return; }
    runtime = r.value();
  }
  std::shared_ptr<BackendRuntime> runtime;
};

TEST_F(OptimizerFactoryTest, UnknownTypeThrows) {
    EXPECT_THROW(
        tinygs::create_optimizer("unknown_optimizer", *runtime, nullptr, nullptr),
        std::runtime_error
    );
}

TEST_F(OptimizerFactoryTest, InvalidTypeVariantsThrow) {
    constexpr std::array<const char*, 11> invalid_types = {
        "",
        "invalid_type",
        "adamm",
        "UNKNOWN",
        "sgd",
        "adamw",
        "adam per gaussian",
        "adam123",
        "adam@per#gaussian",
        "  adam",
        "adam  "
    };

    for (const char* type : invalid_types) {
        EXPECT_THROW(
            tinygs::create_optimizer(type, *runtime, nullptr, nullptr),
            std::runtime_error
        ) << "Expected throw for type: " << type;
    }
}

TEST_F(OptimizerFactoryTest, ErrorMessageContainsType) {
    try {
        tinygs::create_optimizer("test_optimizer", *runtime, nullptr, nullptr);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("test_optimizer") != std::string::npos);
    }
}

TEST_F(OptimizerFactoryTest, ErrorMessageContainsSupportedTypes) {
    try {
        tinygs::create_optimizer("invalid", *runtime, nullptr, nullptr);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("adam") != std::string::npos);
    }
}

} // namespace

