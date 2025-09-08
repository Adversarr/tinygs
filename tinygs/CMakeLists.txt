# Source files (following tiny-cuda-nn patterns)
file(GLOB_RECURSE TINYGS_SOURCES
    "src/*.cpp"
    "src/*.cu"
    "src/*.cuh"
    "src/*.hpp"
)

file(GLOB_RECURSE TINYGS_HEADERS
    "include/*.hpp"
    "include/*.cuh"
    "include/*.h"
)

# Separate CUDA sources for specific handling
file(GLOB_RECURSE TINYGS_CUDA_SOURCES
    "src/*.cu"
)

file(GLOB_RECURSE TINYGS_CPP_SOURCES
    "src/*.cpp"
)

# Create main library (following tiny-cuda-nn patterns)
add_library(tinygs STATIC ${TINYGS_SOURCES} ${TINYGS_HEADERS})

# Set library alias for consistency
add_library(tinygs::tinygs ALIAS tinygs)

# Set target properties (enhanced following tiny-cuda-nn patterns)
set_target_properties(tinygs PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Link libraries (organized following tiny-cuda-nn patterns)
target_link_libraries(tinygs
    PUBLIC
        # CUDA libraries
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        CUDA::cufft
        # System libraries
        Threads::Threads
        # Third-party libraries
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        cxxopts::cxxopts
        OpenMP::OpenMP_CXX
        glm::glm
        # OpenCV
        ${OpenCV_LIBS}
)

target_compile_definitions(
        tinygs
        PUBLIC
        SPDLOG_ACTIVE_LEVEL=0
)

if(TARGET stb)
    target_link_libraries(tinygs PUBLIC stb)
    message(STATUS "Linking stb library to tinygs")
endif()

# Include directories for target (enhanced following tiny-cuda-nn patterns)
target_include_directories(tinygs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        # Add CUDA include directories
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        # Add internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinygs
        # Add OpenCV include directories
        ${OpenCV_INCLUDE_DIRS}
)

# Compiler definitions (following tiny-cuda-nn patterns)
target_compile_definitions(tinygs
    PUBLIC
        TINYGS_MIN_GPU_ARCH=${MIN_GPU_ARCH}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_BFLOAT16_CONVERSIONS__>
)

# CUDA-specific compile options
if(TINYGS_CUDA_SOURCES)
    set_source_files_properties(${TINYGS_CUDA_SOURCES} PROPERTIES
        COMPILE_FLAGS "${CMAKE_CUDA_FLAGS}"
    )
endif()
