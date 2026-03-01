# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build default targets (config_train)
./build.sh

# Build with specific configuration
BUILD_TYPE=Debug ./build.sh

# Build specific targets
TARGETS="config_train" ./build.sh

# Build with specific CUDA architecture
TINYGS_CUDA_ARCHITECTURES="89" ./build.sh
```

The build script handles CMake configuration and compilation. Built executables are copied to the repository root.

## Running Training

```bash
# Train with a config file
./config_train -c configs/garden.json

# Train with visualization
./config_train -c configs/garden.json -v

# Debug logging
./config_train -c configs/garden.json -d
```

## Architecture Overview

tinygs is a C++/CUDA library for 3D Gaussian Splatting. The codebase follows a modular, plugin-style architecture centered around the `Orchestrator` class.

### Core Data Structures

- **`GPUGaussian3d`** (`tinygs/include/tinygs/core/gpu_gaussian.hpp`): SoA structure storing Gaussian primitives on GPU (means, opacities, rotations, scales, SH coefficients). Uses thrust::device_vector.

- **`RasterizeContext`** (`tinygs/include/tinygs/rasterizer/rasterizer.hpp`): Holds forward/backward pass data including GPUBatchInputOutput structures for camera data and rendered outputs.

### Training Pipeline

The `Orchestrator` class coordinates training by composing these components:

1. **DataLoader** → provides training batches (camera poses, images)
2. **Rasterizer** → renders Gaussians to images (forward) and computes gradients (backward)
3. **Optimizer** → updates Gaussian parameters from gradients
4. **Strategy** → handles densification (split/clone/prune Gaussians)
5. **Loss** → computes training losses (L1, SSIM, etc.)
6. **PoseOpt** → optional camera pose refinement

All components are created via factory functions (`create_rasterizer()`, `create_optimizer()`, etc.) and configured through JSON.

### Configuration

Training is driven by JSON config files (see `configs/garden.json`). Each component has a `type` field for factory selection and additional parameters consumed via `set_params()`.

## Key Module Locations

| Module | Headers | Implementation |
|--------|---------|----------------|
| Core (Gaussian, Camera) | `include/tinygs/core/` | `src/core/` |
| CUDA utilities | `include/tinygs/cuda/` | `src/cuda/` |
| Dataloader | `include/tinygs/dataloader/` | `src/dataloader/` |
| Dataset | `include/tinygs/dataset/` | `src/dataset/` |
| Initialization | `include/tinygs/initialization/` | `src/initialization/` |
| Loss functions | `include/tinygs/loss/` | `src/loss/` |
| Optimizers | `include/tinygs/optim/` | `src/optim/` |
| Pose optimization | `include/tinygs/pose_opt/` | `src/pose_opt/` |
| Rasterizers | `include/tinygs/rasterizer/` | `src/rasterizer/` |
| Strategies | `include/tinygs/strategy/` | `src/strategy/` |
| Utilities | `include/tinygs/utils/` | `src/utils/` |

### Rasterizer Implementations

- **default**: Standard 3DGS rasterization
- **fastgs**: Optimized rasterizer with custom forward/backward kernels
- **fastgs_ours/**: Custom FastGS variant
- **fastgs_ours_fp16/**: FP16-optimized variant

## Code Style

```cpp
// Headers: .hpp, implementations: .cu or .cpp, device-only: .cuh
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
void launch_forward(const GPUGaussian3d& gaussians, cudaStream_t stream) {
  const int blocks = (gaussians.size() + 255) / 256;
  forward_kernel<<<blocks, 256, 0, stream>>>(/* args */);
}
```

### Conventions

- Format: `.clang-format` (Google-based, 120 columns)
- Lint: `.clang-tidy` (modern C++, performance checks)
- Naming: `snake_case` for functions/variables, `CamelCase` for types
- Member variables: `m_` prefix for private/protected (e.g., `m_data`)
- CUDA streams: pass explicitly for async operations
- Logging: use `log_info`, `log_warning`, `log_error` from `common.hpp` (never `std::cout` in library code)

## Dependencies

- CUDA 12.4+ (12.6-12.9 recommended)
- CMake 3.28+
- GCC 11+ (C++20 support required)
- OpenCV, spdlog, nlohmann_json, glm, OpenMP, NVTX3

Dependencies are fetched via CMake during configuration (see `cmake/dependencies.cmake`).

## Change Management

- Run `./build.sh` after modifying C++/CUDA code
- Run format check before committing
- Test training before submitting: `./config_train -c configs/garden.json`

### Ask Before

- Adding new third-party dependencies
- Modifying CMake configuration
- Adding new rasterizer implementations
- Changing training defaults in JSON configs

### Never

- Modify files in `cmake/CPM.cmake` or fetched dependencies
- Commit build artifacts (`build/`, `*.ply`, `*.pt`)
- Add Python code (no Python interface yet)
- Modify `.clang-format` or `.clang-tidy` without discussion