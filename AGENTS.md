# AGENTS.md

## General Personality
Code style: concise, precise, clean, clear, extensible. Write HIGH-VERBOSITY code with comments for human review. Make patches and new features surgical (minimal, targeted changes).
Plan and document style: detailed and explicit.
Thorough tests: cover edge cases, error paths, and typical usage for all new functionalities.

## Purpose and scope

Agents help develop tinygs, a C++/CUDA library for 3D Gaussian Splatting. Focus on CUDA kernels, training pipelines, and rasterization. Do not modify third-party dependencies in cmake/.

## Commands (copy/paste, include flags)

```bash
# Build default targets (video_to_png, config_train)
./build.sh

# Build with specific configuration
BUILD_TYPE=Debug ./build.sh

# Build with specific CUDA architecture
TINYGS_CUDA_ARCHITECTURES="89" ./build.sh

# Build specific targets
TARGETS="config_train" ./build.sh

# Train with a config file
./config_train -c configs/garden.json

# Train with visualization
./config_train -c configs/garden.json -v

# Debug logging
./config_train -c configs/garden.json -d
```

## Tech stack

- Language: C++20, CUDA (`.cu`, `.cuh`)
- CMake: 3.28+
- CUDA: 12.4+ (12.6-12.9 recommended)
- GCC: 11+ (C++20 support required)
- Dependencies: OpenCV, spdlog, nlohmann_json, glm, OpenMP, NVTX3

## Repo map

| Path | Purpose |
|------|---------|
| `tinygs/include/tinygs/` | Public headers (core, cuda, rasterizer, optim, etc.) |
| `tinygs/src/` | Implementation files (.cpp, .cu) |
| `examples/` | Executables (config_train, video_to_png, single_gs) |
| `configs/` | JSON training configurations |
| `cmake/` | CMake modules and dependency fetching |
| `build/` | Build artifacts (generated) |

### Key modules (`tinygs/src`)

#### `core/` - Core data structures
- `gaussian.cpp`, `gpu_gaussian.cu` - GPUGaussian3d (SoA layout), CPU/GPU Gaussian primitives
- `camera.cpp`, `camera_ext.cpp`, `camera_loader.cpp` - Camera model, extrinsics, dataset loading
- `pointcloud.cpp` - Point cloud utilities for initialization

#### `cuda/` - CUDA utilities
- `common_host.cu` - Host-side CUDA helpers, device properties
- `reduce.cu` - Parallel reduction kernels (sum, max, etc.)
- `stat.cu` - GPU statistics, mean/variance computation

#### `dataloader/` - Training data loading
- `dataloader.cu` - Base dataloader interface, factory
- `async.cpp` - Asynchronous multi-stream dataloader (prefetch)
- `simple.cpp` - Synchronous single-stream dataloader

#### `dataset/` - Dataset formats
- `dataset.cpp` - Base dataset interface, factory
- `png_folder.cpp` - Image folder dataset (PNG/JPG)
- `video.cpp` - Video file dataset (MP4, etc.)

#### `initialization/` - Gaussian initialization
- `initialization.cpp` - Initialization factory, coord transformation
- `knn.cpp` - K-nearest neighbors for scale estimation
- `random.cpp` - Random initialization strategies

#### `loss/` - Loss functions
- `loss.cu` - Loss factory, combined loss interface
- `l1.cu`, `l2.cu`, `huber.cu` - Basic losses
- `fused_ssim.cu` - Fused SSIM+L1 loss (primary training loss)
- `psnr.cu` - PSNR metric (evaluation only)

#### `optim/` - Optimizers
- `optim.cu` - Optimizer factory, base interface
- `adamw.cu`, `simple_adam.cu` - Adam variants (primary: adamw)
- `lion.cu`, `lamb.cu`, `adan.cu` - Alternative optimizers
- `sgd.cu` - Stochastic gradient descent
- `lr_scheduler.cpp` - Learning rate scheduling (cosine, step, exponential)

#### `pose_opt/` - Camera pose optimization
- `pose_opt.cpp` - Pose optimizer factory, base interface
- `adamw.cpp`, `sgdm.cpp` - Pose optimization strategies

#### `random/` - Random number generation
- `multinomial.cu` - Multinomial sampling for densification

#### `rasterizer/` - Forward/backward rasterization kernels
- `rasterizer.cpp` - Rasterizer factory, base interface
- `default.cu` - Reference 3DGS implementation
- `fastgs.cu` - FastGS implementation (faster, approximate)
- `fastgs_ours/` - Custom FastGS variant (forward.cu, backward.cu, kernels_*.cuh)
- `fastgs_ours_fp16/` - FP16-optimized FastGS variant
- `3dgs_accel/` - Accelerated 3DGS with CUDA graphs

#### `strategy/` - Densification strategies
- `strategy.cpp` - Strategy factory, base interface
- `default.cu` - Standard 3DGS densification (clone, split, prune)
- `improved.cu` - Enhanced densification with better heuristics
- `mcmc.cu` - MCMC-based sampling strategy

#### `utils/` - Utility functions
- `file.cpp` - File I/O, path utilities
- `image_format.cu` - GPU image format conversion (RGB, BGR, RGBA)
- `inspect_change.cu` - Change detection for adaptive training
- `scope_timer.cpp` - RAII timing utilities
- `stbi_wrapper.cpp` - STB image loading wrapper

#### Root-level files
- `orchestrator.cu` - Main training loop, epoch management, checkpointing
- `tinygs.cpp` - Library initialization, version info

## Code style example

```cpp
// Headers: .hpp, implementations: .cu or .cpp
// Factory pattern with create_X(type_string) functions
// JSON config via from_json()/to_json()

// CUDA kernel pattern
__global__ void forward_kernel(const float* __restrict__ means,
                                const float* __restrict__ opacities,
                                float* __restrict__ output,
                                int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  // ...
}

// Host function launches kernel
void launch_forward(const GPUGaussian3d& gaussians, 
                    cudaStream_t stream) {
  const int blocks = (gaussians.size() + 255) / 256;
  forward_kernel<<<blocks, 256, 0, stream>>>(/* args */);
}
```

## Standards

- Format: `.clang-format` (Google-based, 120 columns)
- Lint: `.clang-tidy` (modern C++, performance checks)
- Naming: `snake_case` for functions/variables, `CamelCase` for types
- Member variables: `m_` prefix for private/protected (e.g., `m_data`)
- CUDA streams: pass explicitly for async operations
- Logging: use `log_info`, `log_warning`, `log_error` from `common.hpp`

## Change management

- No specific branching/PR rules configured
- Run build after changes: `./build.sh`
- Test training before submitting: `./config_train -c configs/garden.json`

## Dependencies and environment

Dependencies are fetched via CMake (CPM). Requires:
- CUDA 12.4+ installed and `nvcc` in PATH (or set `NVCC=/path/to/nvcc`)
- OpenCV (system package, e.g., `libopencv-dev`)
- GCC 11+ with C++20 support

No environment variables required. Config files specify all runtime parameters.

## Boundaries

### Always
- Run `./build.sh` after modifying C++/CUDA code
- Follow existing factory patterns for new components
- Use `log_*` macros from `common.hpp` for logging
- Pass CUDA streams explicitly to kernel launch functions
- Run format check before committing

### Ask first
- Adding new third-party dependencies
- Changing CUDA architecture defaults
- Modifying CMake configuration
- Adding new rasterizer implementations
- Changing training defaults in JSON configs

### Never
- Modify files in `cmake/CPM.cmake` or fetched dependencies
- Commit build artifacts (`build/`, `*.ply`, `*.pt`)
- Add Python code (no Python interface yet)
- Modify `.clang-format` or `.clang-tidy` without discussion
- Use `std::cout` directly in library code (use spdlog)