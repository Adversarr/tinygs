# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build default targets (video_to_png, config_train)
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

### Key Module Locations

| Module | Headers | Implementation |
|--------|---------|----------------|
| Core (Gaussian, Camera) | `include/tinygs/core/` | `src/core/` |
| CUDA utilities | `include/tinygs/cuda/` | `src/cuda/` |
| Rasterizers | `include/tinygs/rasterizer/` | `src/rasterizer/` |
| Optimizers | `include/tinygs/optim/` | `src/optim/` |
| Strategies | `include/tinygs/strategy/` | `src/strategy/` |
| Loss functions | `include/tinygs/loss/` | `src/loss/` |

### Rasterizer Implementations

- **default**: Standard 3DGS rasterization
- **fastgs**: Optimized rasterizer with custom forward/backward kernels in `src/rasterizer/fastgs_ours/`
- **fastgs_fp16**: FP16 variant in `src/rasterizer/fastgs_ours_fp16/`

### Configuration

Training is driven by JSON config files (see `configs/garden.json`). Each component has a `type` field for factory selection and additional parameters consumed via `set_params()`.

## Dependencies

- CUDA 12.4+ (12.6-12.9 recommended)
- CMake 3.28+
- GCC 11+ (C++20 support required)
- OpenCV, spdlog, nlohmann_json, glm, OpenMP, NVTX3

Dependencies are fetched via CMake during configuration (see `cmake/dependencies.cmake`).

## Code Conventions

- Headers use `.hpp`, CUDA kernels use `.cu`, device-only headers use `.cuh`
- Factory pattern for polymorphic components with `create_X(type_string)` functions
- JSON serialization via `from_json()`/`to_json()` methods on config structs
- CUDA streams passed explicitly for async operations