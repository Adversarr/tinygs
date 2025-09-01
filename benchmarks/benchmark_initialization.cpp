/**
 * @file benchmark_initialization.cpp
 * @brief Initialization benchmarks for tinygs library
 */

#include <benchmark/benchmark.h>
#include <tinygs/common.hpp>

static void BM_TinyGSConstruction(benchmark::State& state) {
    for (auto _ : state) {
        tinygs::TinyGS gs;
        benchmark::DoNotOptimize(gs);
    }
}
BENCHMARK(BM_TinyGSConstruction);

static void BM_TinyGSInitialization(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        tinygs::TinyGS gs;
        state.ResumeTiming();
        
        bool result = gs.initialize();
        benchmark::DoNotOptimize(result);
        
        state.PauseTiming();
        gs.cleanup();
        state.ResumeTiming();
    }
}
BENCHMARK(BM_TinyGSInitialization);

static void BM_TinyGSInitCleanupCycle(benchmark::State& state) {
    tinygs::TinyGS gs;
    
    for (auto _ : state) {
        gs.initialize();
        gs.cleanup();
    }
}
BENCHMARK(BM_TinyGSInitCleanupCycle);

static void BM_VersionQuery(benchmark::State& state) {
    for (auto _ : state) {
        const char* version = tinygs::getVersion();
        benchmark::DoNotOptimize(version);
    }
}
BENCHMARK(BM_VersionQuery);

static void BM_CudaAvailabilityQuery(benchmark::State& state) {
    for (auto _ : state) {
        bool available = tinygs::isCudaAvailable();
        benchmark::DoNotOptimize(available);
    }
}
BENCHMARK(BM_CudaAvailabilityQuery);