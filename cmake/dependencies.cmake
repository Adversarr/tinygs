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
# CPMAddPackage(
#     NAME glm
#     GIT_TAG 1.0.1
#     GITHUB_REPOSITORY g-truc/glm
#     OPTIONS
#       "GLM_BUILD_TESTS OFF"
#       "BUILD_TESTING OFF"
# )

# Print dependency status
message(STATUS "Dependencies loaded successfully:")
message(STATUS "  - spdlog: ${spdlog_VERSION}")
message(STATUS "  - nlohmann_json: ${nlohmann_json_VERSION}")
# message(STATUS "  - glm: ${glm_VERSION}")
message(STATUS "  - cutlass: ${cutlass_VERSION}")