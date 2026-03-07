#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tinygs/strategy/strategy.hpp>
#include <tinygs/platform/backend_build.hpp>
#include <tinygs/platform/runtime_factory.hpp>
#include <tinygs/cuda/common_host.hpp>

namespace {
using namespace tinygs;

struct StrategyFactoryTest : public ::testing::Test {
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

TEST_F(StrategyFactoryTest, UnknownTypeThrows) {
    EXPECT_THROW(
        tinygs::create_strategy("unknown_strategy", *runtime, nullptr, nullptr, nullptr),
        std::runtime_error
    );
}

TEST_F(StrategyFactoryTest, InvalidTypeVariantsThrow) {
    constexpr std::array<const char*, 10> invalid_types = {
        "",
        "invalid_type",
        "defult",
        "UNKNOWN",
        "default strategy",
        "default123",
        "default@strategy",
        "  default",
        "default  ",
        "MCMC_UNKNOWN"
    };

    for (const char* type : invalid_types) {
        EXPECT_THROW(
            tinygs::create_strategy(type, *runtime, nullptr, nullptr, nullptr),
            std::runtime_error
        ) << "Expected throw for type: " << type;
    }
}

TEST_F(StrategyFactoryTest, ErrorMessageContainsType) {
    try {
        tinygs::create_strategy("test_strategy", *runtime, nullptr, nullptr, nullptr);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("test_strategy"), std::string::npos);
    }
}

} // namespace
