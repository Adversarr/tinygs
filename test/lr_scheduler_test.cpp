#include <gtest/gtest.h>
#include <tinygs/optim/lr_scheduler.hpp>
#include <tinygs/optim/optim.hpp>
#include <nlohmann/json.hpp>

namespace {

using namespace tinygs;

class MockRuntime final : public BackendRuntime {
public:
    BackendType backend_type() const noexcept override { return BackendType::Cuda; }
    int device() const noexcept override { return 0; }
    CapabilityProfile capability_profile() const override { return CapabilityProfile{}; }

protected:
    Result<BackendQueue> do_create_queue(const QueueDesc&) override {
        return Result<BackendQueue>::success(nullptr, backend_type(), "create_queue");
    }
    Result<BackendEvent> do_create_event(const EventDesc&) override {
        return Result<BackendEvent>::success(nullptr, backend_type(), "create_event");
    }
    Result<BackendBuffer> do_create_buffer(const BufferDesc&) override {
        return Result<BackendBuffer>::success(nullptr, backend_type(), "create_buffer");
    }
    BackendError do_record_event(BackendQueue&, BackendEvent&) override {
        return backend_success(backend_type(), "record_event");
    }
    BackendError do_wait_event(BackendQueue&, BackendEvent&) override {
        return backend_success(backend_type(), "wait_event");
    }
    BackendError do_synchronize_queue(BackendQueue&) override {
        return backend_success(backend_type(), "synchronize_queue");
    }
    BackendError do_synchronize_event(BackendEvent&) override {
        return backend_success(backend_type(), "synchronize_event");
    }
    BackendError do_synchronize_device() override {
        return backend_success(backend_type(), "synchronize_device");
    }
    BackendError do_copy_buffer(BackendQueue&, BackendBuffer&, size_t, BackendBuffer&, size_t, size_t) override {
        return backend_success(backend_type(), "copy_buffer_async");
    }
    BackendError do_copy_from_host(BackendQueue&, BackendBuffer&, size_t, const void*, size_t) override {
        return backend_success(backend_type(), "copy_from_host_async");
    }
    BackendError do_copy_to_host(BackendQueue&, void*, BackendBuffer&, size_t, size_t) override {
        return backend_success(backend_type(), "copy_to_host_async");
    }
    BackendError do_transfer_raw(BackendQueue&, void*, const void*, size_t, TransferDirection) override {
        return backend_success(backend_type(), "transfer_raw");
    }
    BackendError do_fill_buffer(BackendQueue&, BackendBuffer&, size_t, uint8_t, size_t) override {
        return backend_success(backend_type(), "fill_buffer_async");
    }
};

BackendRuntime& mock_runtime() {
    static MockRuntime runtime;
    return runtime;
}

class MockOptimizer : public OptimizerBase {
public:
    MockOptimizer() : OptimizerBase(mock_runtime(), nullptr, nullptr) {}
    
    void step(float scale, const BackendQueue* queue) override { 
        (void)scale; 
        (void)queue;
    }
    void step(const GroupStepConfig& step_config, const BackendQueue* queue) override {
        (void)step_config;
        (void)queue;
    }
    void reset(int* indices, int num_reset) override { 
        (void)indices;
        (void)num_reset;
    }
    void reorder(uint* indices, BackendQueue* queue) override { 
        (void)indices; 
        (void)queue;
    }
    void reset_opacity(BackendQueue* queue) override { 
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
