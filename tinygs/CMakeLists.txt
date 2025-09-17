# Source files (explicit listing instead of GLOB)
set(TINYGS_SOURCES
    # Core sources
    "src/core/camera.cpp"
    "src/core/camera_loader.cpp"
    "src/core/gaussian.cpp"
    "src/core/gpu_gaussian.cu"
    "src/core/pointcloud.cpp"
    
    # CUDA utilities
    "src/cuda/common_host.cu"
    "src/cuda/reduce.cu"
    "src/cuda/stat.cu"
    
    # Dataloader
    "src/dataloader/dataloader.cu"
    "src/dataloader/simple.cpp"
    
    # Dataset
    "src/dataset/dataset.cpp"
    "src/dataset/png_folder.cpp"
    "src/dataset/video.cpp"
    
    # Initialization
    "src/initialization/initialization.cpp"
    "src/initialization/knn.cpp"
    "src/initialization/nanoflann.hpp"
    "src/initialization/random.cpp"
    
    # Loss functions
    "src/loss/fused_ssim.cu"
    "src/loss/l1.cu"
    "src/loss/loss.cu"
    "src/loss/psnr.cu"
    
    # Optimizers
    "src/optim/adamw.cu"
    "src/optim/lr_scheduler.cpp"
    "src/optim/optim.cu"
    "src/optim/sgd.cu"
    
    # Random utilities
    "src/random/multinomial.cu"
    
    # Rasterizer
    "src/rasterizer/3dgs_accel/backward.cu"
    "src/rasterizer/3dgs_accel/forward.cu"
    "src/rasterizer/3dgs_accel/rasterizer_impl.cu"
    "src/rasterizer/default.cu"
    "src/rasterizer/fastgs.cu"
    "src/rasterizer/fastgs_ours/backward.cu"
    "src/rasterizer/fastgs_ours/forward.cu"
    "src/rasterizer/rasterizer.cpp"
    
    # Strategy
    "src/strategy/default.cu"
    "src/strategy/mcmc.cu"
    "src/strategy/strategy.cpp"
    
    # Main library
    "src/tinygs.cpp"
    
    # Trainer
    "src/orchestrator.cu"
    
    # Utilities
    "src/utils/file.cpp"
    "src/utils/image_format.cpp"
    "src/utils/inspect_change.cu"
    "src/utils/scope_timer.cpp"
    "src/utils/stbi_wrapper.cpp"
)

# Create main library (following tiny-cuda-nn patterns)
add_library(tinygs STATIC ${TINYGS_SOURCES})

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
        # System libraries
        Threads::Threads
        # Third-party libraries
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        glm::glm
        # OpenCV
        ${OpenCV_LIBS}
)

target_compile_definitions(tinygs
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

# Compiler-specific flags (enhanced following tiny-cuda-nn patterns)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:-O3;-DNDEBUG;-march=native;-mtune=native>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:-O0;-g;-Wall;-Wextra;-Wpedantic>>
    )
endif()

# CUDA compiler flags (following tiny-cuda-nn patterns)
if(MSVC)
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
    )
else()
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-float-conversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=unrecognized_gcc_pragma>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

# Essential CUDA flags for modern C++ and performance
target_compile_options(tinygs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
