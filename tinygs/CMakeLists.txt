# Source files (explicit listing instead of GLOB)
set(TINYGS_SOURCES
    # Core sources
    "src/core/camera.cpp"
    "src/core/camera_ext.cpp"
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
    "src/dataloader/async.cpp"
    
    # Dataset
    "src/dataset/dataset.cpp"
    "src/dataset/png_folder.cpp"
    "src/dataset/video.cpp"
    
    # Initialization
    "src/initialization/initialization.cpp"
    "src/initialization/knn.cpp"
    "src/initialization/random.cpp"
    
    # Loss functions
    "src/loss/fused_ssim.cu"
    "src/loss/l1.cu"
    "src/loss/l2.cu"
    "src/loss/huber.cu"
    "src/loss/loss.cu"
    "src/loss/psnr.cu"
    
    # Optimizers
    "src/optim/adamw.cu"
    "src/optim/simple_adam.cu"
    "src/optim/lion.cu"
    "src/optim/lamb.cu"
    "src/optim/adan.cu"
    "src/optim/lr_scheduler.cpp"
    "src/optim/optim.cu"
    "src/optim/sgd.cu"

    # Pose Optimizers
    "src/pose_opt/pose_opt.cpp"
    "src/pose_opt/sgdm.cpp"
    "src/pose_opt/adamw.cpp"

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
    # "src/rasterizer/gsplat.cu"
    # "src/rasterizer/gsplat/backward.cu"
    # "src/rasterizer/gsplat/forward.cu"
    "src/rasterizer/rasterizer.cpp"
    
    # Strategy
    "src/strategy/default.cu"
    "src/strategy/mcmc.cu"
    "src/strategy/improved.cu"
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

# Create main library
add_library(tinygs STATIC ${TINYGS_SOURCES})
add_library(tinygs::tinygs ALIAS tinygs)

# Standards and CUDA properties
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
        nvtx3-cpp
        # OpenCV
        ${OpenCV_LIBS}
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

# SPDLOG_LEVEL_TRACE    0
# SPDLOG_LEVEL_DEBUG    1
# SPDLOG_LEVEL_INFO     2

# Config-dependent definitions
# - Define NDEBUG for Release/RelWithDebInfo (available to both NVCC and host)
# - Set spdlog compile-time log level based on configuration
target_compile_definitions(tinygs
    PUBLIC
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:NDEBUG>
        $<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=0>
        $<$<CONFIG:RelWithDebInfo>:SPDLOG_ACTIVE_LEVEL=1>
        $<$<CONFIG:Release>:SPDLOG_ACTIVE_LEVEL=2>
)

# MSVC/Non-MSVC specific options
if(MSVC)
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
    )
    # Note: Add /DNOMINMAX in definitions if needed
else()
    if (TINYGS_ENABLE_NATIVE)
        target_compile_options(tinygs PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-march=native>
            $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mtune=native>
        )
    endif()

    # Host-side warning/aliasing rules (native CXX; passed to host compiler via -Xcompiler for CUDA)
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-float-conversion>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing>

        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-float-conversion>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>

        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=unrecognized_gcc_pragma>
    )
endif()

# CUDA-specific options
target_compile_options(tinygs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if (TINYGS_ENABLE_FAST_MATH)
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        # Note: For host-side fast math (risky): $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
    )
endif()

# CUDA device-side feature toggles (change to PUBLIC if common headers depend on these switches)
target_compile_definitions(tinygs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_BFLOAT16_CONVERSIONS__>
)

if (TINYGS_ENABLE_PROFILE)
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endif()

if (TINYGS_ENABLE_NVTX_SYNC)
    target_compile_definitions(tinygs PUBLIC TINYGS_NVTX_SYNC=1)
endif()

# Other public defs
target_compile_definitions(tinygs PUBLIC "TINYGS_MIN_GPU_ARCH=${MIN_GPU_ARCH}")
