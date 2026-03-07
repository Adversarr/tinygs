set(TINYGS_COMMON_SOURCES
    "src/core/camera.cpp"
    "src/core/camera_ext.cpp"
    "src/core/camera_loader.cpp"
    "src/core/pointcloud.cpp"
    "src/dataset/dataset.cpp"
    "src/initialization/initialization.cpp"
    "src/optim/lr_scheduler.cpp"
    "src/pose_opt/pose_opt.cpp"
    "src/pose_opt/adamw.cpp"
    "src/platform/runtime_factory.cpp"
    "src/rasterizer/rasterizer.cpp"
    "src/rasterizer/cpu.cpp"
    "src/strategy/strategy.cpp"
    "src/tinygs.cpp"
    "src/utils/file.cpp"
    "src/utils/scope_timer.cpp"
    "src/utils/stbi_wrapper.cpp"
)

set(TINYGS_BACKEND_CUDA_SOURCES
    "src/core/gpu_gaussian.cu"
    "src/cuda/common_host.cu"
    "src/cuda/reduce.cu"
    "src/dataloader/dataloader.cu"
    "src/dataloader/simple.cpp"
    "src/dataloader/async.cpp"
    "src/dataset/image.cpp"
    "src/initialization/knn.cpp"
    "src/initialization/random.cpp"
    "src/loss/fused_ssim.cu"
    "src/loss/l1.cu"
    "src/loss/l2.cu"
    "src/loss/huber.cu"
    "src/loss/loss.cu"
    "src/loss/psnr.cu"
    "src/optim/adam.cu"
    "src/optim/adam_per_gaussian.cu"
    "src/optim/optim.cu"
    "src/random/multinomial.cu"
    "src/rasterizer/fastgs.cu"
    "src/rasterizer/fastgs_ours/backward.cu"
    "src/rasterizer/fastgs_ours/forward.cu"
    "src/rasterizer/fastgs_ours_fp16/backward.cu"
    "src/rasterizer/fastgs_ours_fp16/forward.cu"
    "src/strategy/absgs.cu"
    "src/strategy/default.cu"
    "src/strategy/fastgs.cu"
    "src/strategy/mcmc.cu"
    "src/strategy/improved.cu"
    "src/orchestrator.cpp"
    "src/utils/image_format.cu"
)

set(TINYGS_BACKEND_CUDA_RUNTIME_SOURCES
    "src/cuda/runtime_cuda.cpp"
)

set(TINYGS_METAL_RUNTIME_SOURCES
    "src/platform/runtime_metal.mm"
)

set(TINYGS_BACKEND_METAL_SOURCES
)

set(TINYGS_SOURCES ${TINYGS_COMMON_SOURCES})
if(TINYGS_BACKEND STREQUAL "CUDA")
    list(APPEND TINYGS_SOURCES ${TINYGS_BACKEND_CUDA_RUNTIME_SOURCES})
    list(APPEND TINYGS_SOURCES ${TINYGS_BACKEND_CUDA_SOURCES})
elseif(TINYGS_BACKEND STREQUAL "METAL")
    list(APPEND TINYGS_SOURCES ${TINYGS_METAL_RUNTIME_SOURCES})
    list(APPEND TINYGS_SOURCES ${TINYGS_BACKEND_METAL_SOURCES})
else()
    message(FATAL_ERROR "Backend ${TINYGS_BACKEND} is not wired in tinygs/CMakeLists.txt yet.")
endif()

if(TINYGS_BACKEND STREQUAL "METAL" AND NOT TINYGS_BACKEND_METAL_SOURCES)
    message(STATUS "No Metal module sources listed yet; building Metal runtime placeholder only.")
endif()

# Create main library
add_library(tinygs STATIC ${TINYGS_SOURCES})
add_library(tinygs::tinygs ALIAS tinygs)

# Standards and CUDA properties
set_target_properties(tinygs PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(TINYGS_BACKEND STREQUAL "CUDA")
    set_target_properties(tinygs PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

# Link libraries (organized following tiny-cuda-nn patterns)
target_link_libraries(tinygs
    PUBLIC
        # System libraries
        Threads::Threads
        # Third-party libraries
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        glm::glm
        nvtx3-cpp
        # OpenCV
        ${OpenCV_LIBS}
)

if(TINYGS_BACKEND STREQUAL "CUDA")
    target_link_libraries(tinygs
        PUBLIC
            CUDA::cuda_driver
            CUDA::cudart
            OpenMP::OpenMP_CXX
    )
elseif(TINYGS_BACKEND STREQUAL "METAL")
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(CORE_GRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
    
    target_link_libraries(tinygs
        PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${CORE_GRAPHICS_FRAMEWORK}
    )
elseif(OpenMP_FOUND)
    target_link_libraries(tinygs PUBLIC OpenMP::OpenMP_CXX)
endif()


if(TARGET stb)
    target_link_libraries(tinygs PUBLIC stb)
    message(STATUS "Linking stb library to tinygs")
endif()

# Include directories for target (enhanced following tiny-cuda-nn patterns)
target_include_directories(tinygs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        # Add internal headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tinygs
        # Add OpenCV include directories
        ${OpenCV_INCLUDE_DIRS}
)

if(TINYGS_BACKEND STREQUAL "CUDA")
    target_include_directories(tinygs
        PUBLIC
            ${CUDAToolkit_INCLUDE_DIRS}
            $<$<VERSION_GREATER_EQUAL:${CUDA_VERSION},13.0>:${CUDAToolkit_INCLUDE_DIRS}/cccl>
    )
elseif(TINYGS_BACKEND STREQUAL "METAL")
    set_source_files_properties(
        ${TINYGS_METAL_RUNTIME_SOURCES}
        PROPERTIES
            LANGUAGE OBJCXX
            COMPILE_FLAGS "-fobjc-arc"
    )
endif()

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
        )
        if(TINYGS_BACKEND STREQUAL "CUDA")
            target_compile_options(tinygs PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mtune=native>
            )
        endif()
    endif()

    # Host-side warning/aliasing rules (native CXX; passed to host compiler via -Xcompiler for CUDA)
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-float-conversion>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing>
    )
    if(TINYGS_BACKEND STREQUAL "CUDA")
        target_compile_options(tinygs PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-float-conversion>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=unrecognized_gcc_pragma>
        )
    endif()
endif()

# CUDA-specific options
if(TINYGS_BACKEND STREQUAL "CUDA")
    target_compile_options(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

if (TINYGS_ENABLE_FAST_MATH)
    if(TINYGS_BACKEND STREQUAL "CUDA")
        target_compile_options(tinygs PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            # Note: For host-side fast math (risky): $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
        )
    endif()
endif()

# CUDA device-side feature toggles (change to PUBLIC if common headers depend on these switches)
if(TINYGS_BACKEND STREQUAL "CUDA")
    target_compile_definitions(tinygs PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_BFLOAT16_CONVERSIONS__>
    )
endif()


# Manually set NDEBUG if Release/RelWithDebInfo
target_compile_definitions(tinygs
    PRIVATE
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:NDEBUG>
)



if (TINYGS_ENABLE_PROFILE)
    if(TINYGS_BACKEND STREQUAL "CUDA")
        target_compile_options(tinygs PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
    endif()
endif()

if (TINYGS_ENABLE_NVTX_SYNC)
    target_compile_definitions(tinygs PUBLIC TINYGS_NVTX_SYNC=1)
endif()

# Other public defs
target_compile_definitions(tinygs PUBLIC "TINYGS_MIN_GPU_ARCH=${MIN_GPU_ARCH}")
