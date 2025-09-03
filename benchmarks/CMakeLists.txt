# Benchmarks CMakeLists.txt for tinygs

# Basic benchmarks
add_executable(tinygs_benchmarks
    benchmark_main.cpp
    benchmark_dataloader.cpp
    benchmark_fastgs.cpp
)

# Link libraries
target_link_libraries(tinygs_benchmarks
    tinygs::tinygs
    benchmark::benchmark
)
