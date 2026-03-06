cmake_minimum_required(VERSION 3.28)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TINYGS_SUPPORTED_BACKENDS CUDA HIP METAL)
set(TINYGS_BACKEND "CUDA" CACHE STRING "Backend to build (CUDA|HIP|METAL)")
set_property(CACHE TINYGS_BACKEND PROPERTY STRINGS ${TINYGS_SUPPORTED_BACKENDS})
string(TOUPPER "${TINYGS_BACKEND}" TINYGS_BACKEND)
if(NOT TINYGS_BACKEND IN_LIST TINYGS_SUPPORTED_BACKENDS)
    message(FATAL_ERROR
        "Invalid TINYGS_BACKEND='${TINYGS_BACKEND}'. "
        "Allowed values: CUDA, HIP, METAL."
    )
endif()

set(TINYGS_PROJECT_LANGUAGES CXX)
if(TINYGS_BACKEND STREQUAL "CUDA")
    list(APPEND TINYGS_PROJECT_LANGUAGES CUDA)
endif()

# Project configuration
project(tinygs
        VERSION 1.0.0
        DESCRIPTION "Lightweight library for reconstructing 3D scenes with 3DGS in seconds"
        LANGUAGES ${TINYGS_PROJECT_LANGUAGES}
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(TINYGS_BACKEND STREQUAL "CUDA")
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Platform check
if(NOT UNIX)
    message(FATAL_ERROR "tinygs is currently only supported on Unix-like platforms")
endif()
option(TINYGS_BUILD_APPS "Build tinygs application binaries" ON)
option(TINYGS_BUILD_TESTS "Build tinygs test binaries" OFF)
option(TINYGS_ENABLE_PROFILE "Enable profiling with lineinfo" OFF)
option(TINYGS_ENABLE_NVTX_SYNC "Enable operation sync to get accurate nvtx range" OFF)
option(TINYGS_ENABLE_NATIVE "Enable native optimizations" ON)
option(TINYGS_ENABLE_FAST_MATH "Enable fast math optimizations" ON)
add_compile_definitions(TINYGS_BACKEND_${TINYGS_BACKEND}=1)

set(MIN_GPU_ARCH 0)
if(TINYGS_BACKEND STREQUAL "CUDA")
    if(NOT DEFINED TINYGS_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "86")
        message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES ${TINYGS_CUDA_ARCHITECTURES})
        message(STATUS "Using user-defined CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Set minimum GPU architecture definition (following tiny-cuda-nn pattern)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)
endif()
add_compile_definitions(TINYGS_MIN_GPU_ARCH=${MIN_GPU_ARCH})

### Dependencies
include(${PROJECT_SOURCE_DIR}/cmake/dependencies.cmake)

### library

# Find required packages
find_package(Threads REQUIRED)
if(TINYGS_BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP REQUIRED)
else()
    find_package(OpenMP QUIET)
endif()

# Figure out CUDA version
if(TINYGS_BACKEND STREQUAL "CUDA" AND CMAKE_CUDA_COMPILER_LOADED)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
        set(CUDA_VERSION "${CMAKE_MATCH_1}")
    endif()
endif()

# CUDA version validation (following tiny-cuda-nn pattern)
if(TINYGS_BACKEND STREQUAL "CUDA" AND CUDA_VERSION VERSION_LESS 12.4)
    message(FATAL_ERROR "CUDA version too low. tinygs requires CUDA 12.4 or higher. Found: ${CUDA_VERSION}")
endif()

add_subdirectory(tinygs)

# Optional: Build application binaries
if(TINYGS_BUILD_APPS)
    if(TINYGS_BACKEND STREQUAL "CUDA")
        add_subdirectory(apps)
    else()
        message(STATUS "Skipping apps for backend ${TINYGS_BACKEND}: CUDA-only app sources still need backend-neutralization.")
    endif()
endif()

if(TINYGS_BUILD_TESTS)
    if(TINYGS_BACKEND STREQUAL "CUDA")
        enable_testing()
        add_subdirectory(test)
    else()
        message(STATUS "Skipping tests for backend ${TINYGS_BACKEND}: current test targets still depend on CUDA-only sources.")
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "tinygs Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Backend: ${TINYGS_BACKEND}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
if(TINYGS_BACKEND STREQUAL "CUDA")
    message(STATUS "  CUDA standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "  CUDA version: ${CUDA_VERSION}")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "  CUDA standard: disabled")
    message(STATUS "  CUDA version: n/a")
    message(STATUS "  CUDA architectures: n/a")
endif()
message(STATUS "  Minimum GPU architecture: ${MIN_GPU_ARCH}")
message(STATUS "  Build apps: ${TINYGS_BUILD_APPS}")
message(STATUS "  Build tests: ${TINYGS_BUILD_TESTS}")
message(STATUS "")
