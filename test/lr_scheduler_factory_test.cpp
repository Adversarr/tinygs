#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tinygs/optim/lr_scheduler.hpp>

namespace {

using namespace tinygs;

TEST(LrSchedulerFactoryTest, UnknownTypeThrows) {
    EXPECT_THROW(
        create_lr_scheduler("unknown_scheduler", nullptr, OptimParamGroup::Means),
        std::invalid_argument
    );
}

TEST(LrSchedulerFactoryTest, InvalidTypeVariantsThrow) {
    constexpr std::array<const char*, 12> invalid_types = {
        "",
        "invalid_type",
        "constat",
        "exponental",
        "UNKNOWN",
        "step",
        "cosine",
        "constant lr",
        "constant1",
        "constant@lr",
        "constant_lr",
        " constant"
    };

    for (const char* type : invalid_types) {
        EXPECT_THROW(
            create_lr_scheduler(type, nullptr, OptimParamGroup::Means),
            std::invalid_argument
        ) << "Expected throw for type: " << type;
    }
}

TEST(LrSchedulerFactoryTest, ErrorMessageContainsType) {
    try {
        create_lr_scheduler("test_scheduler", nullptr, OptimParamGroup::Means);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("test_scheduler") != std::string::npos);
    }
}

TEST(LrSchedulerFactoryTest, ErrorMessageContainsUnknownWord) {
    try {
        create_lr_scheduler("invalid", nullptr, OptimParamGroup::Means);
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Unknown") != std::string::npos);
    }
}

TEST(LrSchedulerFactoryTest, CreateConstantWithNullptrOptimizer) {
    auto scheduler = create_lr_scheduler("constant", nullptr, OptimParamGroup::Means);
    EXPECT_NE(scheduler, nullptr);
    EXPECT_NO_THROW(scheduler->step());
}

TEST(LrSchedulerFactoryTest, CreateExponentialWithNullptrOptimizer) {
    auto scheduler = create_lr_scheduler("exponential", nullptr, OptimParamGroup::Means);
    EXPECT_NE(scheduler, nullptr);
    EXPECT_NO_THROW(scheduler->step());
}

TEST(LrSchedulerFactoryTest, CreateLowercase) {
    auto scheduler1 = create_lr_scheduler("constant", nullptr, OptimParamGroup::Means);
    EXPECT_NE(scheduler1, nullptr);
    
    auto scheduler2 = create_lr_scheduler("exponential", nullptr, OptimParamGroup::Means);
    EXPECT_NE(scheduler2, nullptr);
}

TEST(LrSchedulerFactoryTest, CreateCaseInsensitiveUppercase) {
    auto scheduler1 = create_lr_scheduler("CONSTANT", nullptr, OptimParamGroup::Means);
    EXPECT_NE(scheduler1, nullptr);
    
    auto scheduler2 = create_lr_scheduler("EXPONENTIAL", nullptr, OptimParamGroup::Means);
    EXPECT_NE(scheduler2, nullptr);
}

TEST(LrSchedulerFactoryTest, DifferentOptimParamGroups) {
    auto scheduler_means = create_lr_scheduler("constant", nullptr, OptimParamGroup::Means);
    auto scheduler_shs = create_lr_scheduler("constant", nullptr, OptimParamGroup::Shs);
    auto scheduler_opacities = create_lr_scheduler("constant", nullptr, OptimParamGroup::Opacities);
    auto scheduler_scales = create_lr_scheduler("constant", nullptr, OptimParamGroup::Scales);
    auto scheduler_rotations = create_lr_scheduler("constant", nullptr, OptimParamGroup::Rotations);
    
    EXPECT_NE(scheduler_means, nullptr);
    EXPECT_NE(scheduler_shs, nullptr);
    EXPECT_NE(scheduler_opacities, nullptr);
    EXPECT_NE(scheduler_scales, nullptr);
    EXPECT_NE(scheduler_rotations, nullptr);
}

}
