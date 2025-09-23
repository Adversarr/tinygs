cmake_minimum_required(VERSION 3.21)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Platform check
if(NOT UNIX)
    message(FATAL_ERROR "tinygs is currently only supported on Linux platforms")
endif()

if(NOT DEFINED TINYGS_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(CMAKE_CUDA_ARCHITECTURES ${TINYGS_CUDA_ARCHITECTURES})
    message(STATUS "Using user-defined CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set minimum GPU architecture definition (following tiny-cuda-nn pattern)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)
add_compile_definitions(TINYGS_MIN_GPU_ARCH=${MIN_GPU_ARCH})


# Project configuration
project(tinygs
        VERSION 1.0.0
        DESCRIPTION "Lightweight library for reconstructing 3D scenes with 3DGS in seconds"
        LANGUAGES CXX CUDA
)
option(TINYGS_BUILD_EXAMPLES "Build tinygs examples" ON)
option(TINYGS_BUILD_TESTS "Build tinygs tests" OFF)
option(TINYGS_BUILD_BENCHMARKS "Build tinygs benchmarks" ON)
option(TINYGS_ENABLE_PROFILE "Enable profiling with lineinfo" OFF)
option(TINYGS_ENABLE_NVTX_SYNC "Enable operation sync to get accurate nvtx range" OFF)
option(TINYGS_ENABLE_NATIVE "Enable native optimizations" ON)
option(TINYGS_ENABLE_FAST_MATH "Enable fast math optimizations" ON)

### Dependencies
include(${PROJECT_SOURCE_DIR}/cmake/dependencies.cmake)

### library

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Figure out CUDA version
if(CMAKE_CUDA_COMPILER_LOADED)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
        set(CUDA_VERSION "${CMAKE_MATCH_1}")
    endif()
endif()

# CUDA version validation (following tiny-cuda-nn pattern)
if(CUDA_VERSION VERSION_LESS 10.2)
    message(FATAL_ERROR "CUDA version too low. tinygs requires CUDA 10.2 or higher. Found: ${CUDA_VERSION}")
endif()

add_subdirectory(tinygs)

# Optional: Build examples
if(TINYGS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optional: Build tests
if(TINYGS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Optional: Build benchmarks
if(TINYGS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "tinygs Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA version: ${CUDA_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Minimum GPU architecture: ${MIN_GPU_ARCH}")
message(STATUS "  Build examples: ${TINYGS_BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${TINYGS_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${TINYGS_BUILD_BENCHMARKS}")
message(STATUS "")