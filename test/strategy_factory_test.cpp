#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tinygs/strategy/strategy.hpp>

TEST(StrategyFactoryTest, UnknownTypeThrows) {
    EXPECT_THROW(
        tinygs::create_strategy("unknown_strategy", nullptr, nullptr, nullptr),
        std::runtime_error
    );
}

TEST(StrategyFactoryTest, InvalidTypeVariantsThrow) {
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
            tinygs::create_strategy(type, nullptr, nullptr, nullptr),
            std::runtime_error
        ) << "Expected throw for type: " << type;
    }
}

TEST(StrategyFactoryTest, ErrorMessageContainsType) {
    try {
        tinygs::create_strategy("test_strategy", nullptr, nullptr, nullptr);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("test_strategy"), std::string::npos);
    }
}
