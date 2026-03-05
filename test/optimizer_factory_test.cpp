#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tinygs/optim/optim.hpp>

TEST(OptimizerFactoryTest, UnknownTypeThrows) {
    EXPECT_THROW(
        tinygs::create_optimizer("unknown_optimizer", nullptr, nullptr, nullptr),
        std::runtime_error
    );
}

TEST(OptimizerFactoryTest, InvalidTypeVariantsThrow) {
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
            tinygs::create_optimizer(type, nullptr, nullptr, nullptr),
            std::runtime_error
        ) << "Expected throw for type: " << type;
    }
}

TEST(OptimizerFactoryTest, ErrorMessageContainsType) {
    try {
        tinygs::create_optimizer("test_optimizer", nullptr, nullptr, nullptr);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("test_optimizer") != std::string::npos);
    }
}

TEST(OptimizerFactoryTest, ErrorMessageContainsSupportedTypes) {
    try {
        tinygs::create_optimizer("invalid", nullptr, nullptr, nullptr);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("adam") != std::string::npos);
    }
}

