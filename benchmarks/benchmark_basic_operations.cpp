/**
 * @file benchmark_basic_operations.cpp
 * @brief Basic operations benchmarks for tinygs library
 */

#include <benchmark/benchmark.h>
#include <tinygs/common.hpp>
#include <memory>
#include <vector>

class TinyGSFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        gs = std::make_unique<tinygs::TinyGS>();
        gs->initialize();
    }
    
    void TearDown(const ::benchmark::State& state) override {
        gs->cleanup();
        gs.reset();
    }
    
    std::unique_ptr<tinygs::TinyGS> gs;
};

BENCHMARK_F(TinyGSFixture, BM_BasicOperation)(benchmark::State& state) {
    for (auto _ : state) {
        // Placeholder for basic operations
        // This would benchmark actual TinyGS operations once implemented
        benchmark::DoNotOptimize(gs.get());
    }
}

static void BM_MultipleInstances(benchmark::State& state) {
    const int num_instances = state.range(0);
    
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<std::unique_ptr<tinygs::TinyGS>> instances;
        instances.reserve(num_instances);
        state.ResumeTiming();
        
        for (int i = 0; i < num_instances; ++i) {
            instances.emplace_back(std::make_unique<tinygs::TinyGS>());
        }
        
        benchmark::DoNotOptimize(instances);
    }
}
BENCHMARK(BM_MultipleInstances)->Range(1, 64);

static void BM_MemoryAllocation(benchmark::State& state) {
    const int allocation_size = state.range(0);
    
    for (auto _ : state) {
        std::vector<uint8_t> buffer(allocation_size);
        benchmark::DoNotOptimize(buffer.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_MemoryAllocation)->Range(1024, 1024*1024*16);