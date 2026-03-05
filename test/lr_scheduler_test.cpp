#include <gtest/gtest.h>
#include <tinygs/optim/lr_scheduler.hpp>
#include <tinygs/optim/optim.hpp>
#include <nlohmann/json.hpp>

namespace {

using namespace tinygs;

class MockOptimizer : public OptimizerBase {
public:
    MockOptimizer() : OptimizerBase(nullptr, nullptr, nullptr) {}
    
    void step(float scale, BackendStream stream) override { 
        (void)scale; 
        (void)stream;
    }
    void step(const GroupStepConfig& step_config, BackendStream stream) override {
        (void)step_config;
        (void)stream;
    }
    void reset(int* indices, int num_reset) override { 
        (void)indices;
        (void)num_reset;
    }
    void reorder(uint* indices, const std::shared_ptr<BackendQueue>& queue) override { 
        (void)indices; 
        (void)queue;
    }
    void reset_opacity(const std::shared_ptr<BackendQueue>& queue) override { 
        (void)queue; 
    }
};

TEST(ConstantLRTest, ReturnsConstantValue) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ConstantLR scheduler(optimizer, OptimParamGroup::Means, 0.5f);
    
    EXPECT_FLOAT_EQ(scheduler.get_lr(), 0.5f);
    EXPECT_FLOAT_EQ(scheduler.step(), 0.5f);
    EXPECT_FLOAT_EQ(scheduler.step(), 0.5f);
    EXPECT_FLOAT_EQ(scheduler.step(), 0.5f);
}

TEST(ConstantLRTest, UpdatesOptimizerLR) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ConstantLR scheduler(optimizer, OptimParamGroup::Means, 0.25f);
    
    EXPECT_FLOAT_EQ(optimizer->get_lr(OptimParamGroup::Means), 0.25f);
    
    scheduler.step();
    EXPECT_FLOAT_EQ(optimizer->get_lr(OptimParamGroup::Means), 0.25f);
}

TEST(ConstantLRTest, ResetDoesNothing) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ConstantLR scheduler(optimizer, OptimParamGroup::Means, 1.0f);
    
    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.get_lr(), 1.0f);
}

TEST(ConstantLRTest, GetParamsReturnsCorrectType) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ConstantLR scheduler(optimizer, OptimParamGroup::Means, 0.75f);
    
    json params = scheduler.get_params();
    EXPECT_EQ(params["type"], "constant");
    EXPECT_FLOAT_EQ(params["lr"].get<float>(), 0.75f);
}

TEST(ConstantLRTest, SetParamsUpdatesLR) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ConstantLR scheduler(optimizer, OptimParamGroup::Means, 1.0f);
    
    json new_params;
    new_params["lr"] = 0.33f;
    scheduler.set_params(new_params);
    
    EXPECT_FLOAT_EQ(scheduler.get_lr(), 0.33f);
    EXPECT_FLOAT_EQ(optimizer->get_lr(OptimParamGroup::Means), 0.33f);
}

TEST(ExponentialLRTest, InitialLRIsSet) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 1.0f, 0.9f);
    
    EXPECT_FLOAT_EQ(scheduler.get_lr(), 1.0f);
}

TEST(ExponentialLRTest, DecaysExponentially) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 1.0f, 0.5f);
    
    float lr0 = scheduler.step();
    float lr1 = scheduler.step();
    float lr2 = scheduler.step();
    
    EXPECT_FLOAT_EQ(lr0, 1.0f * std::pow(0.5f, 0));
    EXPECT_FLOAT_EQ(lr1, 1.0f * std::pow(0.5f, 1));
    EXPECT_FLOAT_EQ(lr2, 1.0f * std::pow(0.5f, 2));
}

TEST(ExponentialLRTest, UpdatesOptimizerLR) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 2.0f, 0.8f);
    
    EXPECT_FLOAT_EQ(optimizer->get_lr(OptimParamGroup::Means), 2.0f);
    
    scheduler.step();
    float expected = 2.0f * std::pow(0.8f, 0);
    EXPECT_FLOAT_EQ(optimizer->get_lr(OptimParamGroup::Means), expected);
}

TEST(ExponentialLRTest, ResetRestoresInitialLR) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 1.0f, 0.5f);
    
    scheduler.step();
    scheduler.step();
    scheduler.step();
    
    scheduler.reset();
    EXPECT_FLOAT_EQ(scheduler.get_lr(), 1.0f);
    EXPECT_FLOAT_EQ(optimizer->get_lr(OptimParamGroup::Means), 1.0f);
}

TEST(ExponentialLRTest, GetParamsReturnsCorrectType) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Shs, 0.5f, 0.95f);
    
    json params = scheduler.get_params();
    EXPECT_EQ(params["type"], "exponential");
    EXPECT_FLOAT_EQ(params["initial_lr"].get<float>(), 0.5f);
    EXPECT_FLOAT_EQ(params["decay_rate"].get<float>(), 0.95f);
}

TEST(ExponentialLRTest, SetParamsUpdatesDecayRate) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 1.0f, 0.9f);
    
    json new_params;
    new_params["decay_rate"] = 0.5f;
    scheduler.set_params(new_params);
    
    json params = scheduler.get_params();
    EXPECT_FLOAT_EQ(params["decay_rate"].get<float>(), 0.5f);
}

TEST(ExponentialLRTest, FastGSScheduleZeroLR) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 0.0f, 0.9f);
    
    json params;
    params["final_lr"] = 0.0f;
    params["use_fastgs_schedule"] = true;
    scheduler.set_params(params);
    
    float lr = scheduler.step();
    EXPECT_FLOAT_EQ(lr, 0.0f);
}

TEST(ExponentialLRTest, FastGSScheduleInterpolation) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 1.0f, 0.9f);
    
    json params;
    params["final_lr"] = 0.1f;
    params["max_steps"] = 100;
    params["use_fastgs_schedule"] = true;
    scheduler.set_params(params);
    
    float lr0 = scheduler.step();
    
    for (int i = 0; i < 99; ++i) {
        scheduler.step();
    }
    float lr100 = scheduler.step();
    
    EXPECT_GT(lr0, lr100);
    EXPECT_GT(lr0, 0.0f);
    EXPECT_GT(lr100, 0.0f);
}

TEST(ExponentialLRTest, FastGSScheduleWithDelay) {
    auto optimizer = std::make_shared<MockOptimizer>();
    ExponentialLR scheduler(optimizer, OptimParamGroup::Means, 1.0f, 0.9f);
    
    json params;
    params["final_lr"] = 0.01f;
    params["delay_mult"] = 0.01f;
    params["delay_steps"] = 100;
    params["max_steps"] = 1000;
    params["use_fastgs_schedule"] = true;
    scheduler.set_params(params);
    
    float lr = scheduler.step();
    EXPECT_GT(lr, 0.0f);
    EXPECT_LT(lr, 1.0f);
}

TEST(CreateLrSchedulerTest, CreatesConstantScheduler) {
    auto optimizer = std::make_shared<MockOptimizer>();
    auto scheduler = create_lr_scheduler("constant", optimizer, OptimParamGroup::Means);
    
    EXPECT_NE(scheduler, nullptr);
    EXPECT_EQ(scheduler->get_params()["type"], "constant");
}

TEST(CreateLrSchedulerTest, CreatesExponentialScheduler) {
    auto optimizer = std::make_shared<MockOptimizer>();
    auto scheduler = create_lr_scheduler("exponential", optimizer, OptimParamGroup::Shs);
    
    EXPECT_NE(scheduler, nullptr);
    EXPECT_EQ(scheduler->get_params()["type"], "exponential");
}

TEST(CreateLrSchedulerTest, CreatesSchedulerCaseInsensitive) {
    auto optimizer = std::make_shared<MockOptimizer>();
    
    auto s1 = create_lr_scheduler("CONSTANT", optimizer, OptimParamGroup::Means);
    auto s2 = create_lr_scheduler("Constant", optimizer, OptimParamGroup::Means);
    auto s3 = create_lr_scheduler("ExPoNeNtIaL", optimizer, OptimParamGroup::Means);
    
    EXPECT_NE(s1, nullptr);
    EXPECT_NE(s2, nullptr);
    EXPECT_NE(s3, nullptr);
}

TEST(CreateLrSchedulerTest, ThrowsOnUnknownType) {
    auto optimizer = std::make_shared<MockOptimizer>();
    
    EXPECT_THROW(create_lr_scheduler("unknown", optimizer, OptimParamGroup::Means), std::invalid_argument);
    EXPECT_THROW(create_lr_scheduler("invalid_type", optimizer, OptimParamGroup::Means), std::invalid_argument);
}

TEST(LrSchedulerBaseTest, HandlesNullOptimizer) {
    std::shared_ptr<OptimizerBase> null_optimizer = nullptr;
    ExponentialLR scheduler(null_optimizer, OptimParamGroup::Means, 1.0f, 0.9f);
    
    EXPECT_NO_THROW(scheduler.step());
}

}
