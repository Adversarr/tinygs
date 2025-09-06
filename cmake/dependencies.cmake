# Dependencies management following tiny-cuda-nn patterns
include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

# Core utility libraries
CPMAddPackage(
  NAME cxxopts
  VERSION 3.3.1
  GITHUB_REPOSITORY jarro2783/cxxopts
  OPTIONS
    "CXXOPTS_BUILD_EXAMPLES OFF"
    "CXXOPTS_BUILD_TESTS OFF"
)

CPMAddPackage(
  NAME spdlog
  VERSION 1.15.3
  GITHUB_REPOSITORY gabime/spdlog
  OPTIONS
    "SPDLOG_BUILD_EXAMPLE OFF"
    "SPDLOG_BUILD_TESTS OFF"
    "SPDLOG_BUILD_BENCH OFF"
)

# JSON library
CPMAddPackage(
  NAME nlohmann_json
  VERSION 3.12.0
  GITHUB_REPOSITORY nlohmann/json
  OPTIONS
    "JSON_BuildTests OFF"
    "JSON_Install ON"
)

# Mathematics library
CPMAddPackage(
 NAME glm
 GIT_TAG 1.0.1
 GITHUB_REPOSITORY g-truc/glm
 OPTIONS
   "GLM_BUILD_TESTS OFF"
   "BUILD_TESTING OFF"
)

if(TINYGS_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        CPMAddPackage(
            NAME benchmark
            GIT_TAG v1.9.4
            GITHUB_REPOSITORY google/benchmark
            OPTIONS
                "BENCHMARK_ENABLE_TESTING OFF"
                "BENCHMARK_ENABLE_INSTALL OFF"
        )
    endif()
endif()
