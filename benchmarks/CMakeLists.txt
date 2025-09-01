# Benchmarks CMakeLists.txt for tinygs

# Find required packages
find_package(tinygs REQUIRED)

# Try to find Google Benchmark
find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    # If benchmark is not found, fetch it
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/v1.8.3.zip
    )
    # Disable benchmark tests
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# Basic benchmarks
add_executable(tinygs_benchmarks
    benchmark_main.cpp
    benchmark_initialization.cpp
    benchmark_basic_operations.cpp
)

# Link libraries
target_link_libraries(tinygs_benchmarks
    tinygs::tinygs
    benchmark::benchmark
    benchmark::benchmark_main
)

# Set properties
set_target_properties(tinygs_benchmarks PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# CUDA benchmarks (if CUDA is available)
if(CMAKE_CUDA_COMPILER)
    add_executable(tinygs_cuda_benchmarks
        benchmark_cuda.cu
        benchmark_main.cpp
    )
    
    target_link_libraries(tinygs_cuda_benchmarks
        tinygs::tinygs
        benchmark::benchmark
        benchmark::benchmark_main
    )
    
    set_target_properties(tinygs_cuda_benchmarks PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# Install benchmarks (optional)
install(TARGETS tinygs_benchmarks
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks
)

if(TARGET tinygs_cuda_benchmarks)
    install(TARGETS tinygs_cuda_benchmarks
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks
    )
endif()