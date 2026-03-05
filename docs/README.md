# tinygs Documentation

tinygs is a lightweight C++/CUDA library for 3D Gaussian Splatting scene reconstruction.

## Table of Contents

1. [Architecture Overview](./ARCHITECTURE.md) - Overall system design and component relationships
2. [CLI Tools](./CLI.md) - Command-line executables and their usage
3. [Configuration](./CONFIGURATION.md) - JSON configuration format and parameters
4. [Multiplatform Roadmap](./MULTIPLATFORM_ROADMAP.md) - Phase-based backend architecture plan
5. [Phase 0 Guardrails](./MULTIPLATFORM_PHASE0_GUARDRAILS.md) - Boundary and layering policy
6. [Phase 1 Runtime Contract](./MULTIPLATFORM_PHASE1_RUNTIME_CONTRACT.md) - Backend runtime interfaces and lifecycle rules
7. [Phase 1 Status](./MULTIPLATFORM_PHASE1_STATUS.md) - Phase gate checklist and follow-ups
8. [Phase 2 Memory and Error Contract](./MULTIPLATFORM_PHASE2_MEMORY_AND_ERROR.md) - Runtime memory semantics and deterministic status rules
9. [Phase 2 Status](./MULTIPLATFORM_PHASE2_STATUS.md) - Phase gate checklist and follow-ups
10. [Modules](./modules/)
   - [Core](./modules/core.md) - Data structures for Gaussians, cameras, and images
   - [CUDA](./modules/cuda.md) - CUDA utilities, memory management, and kernels
   - [Orchestrator](./modules/orchestrator.md) - Main training coordinator and pipeline
   - [Rasterizer](./modules/rasterizer.md) - Forward/backward rendering kernels
   - [Optimizer](./modules/optimizer.md) - Parameter optimization algorithms
   - [Strategy](./modules/strategy.md) - Densification and pruning strategies
   - [Dataloader](./modules/dataloader.md) - Data loading and batching
   - [Dataset](./modules/dataset.md) - Dataset formats and loading
   - [Loss](./modules/loss.md) - Loss functions and metrics
   - [Initialization](./modules/initialization.md) - Gaussian initialization methods
   - [Pose Optimization](./modules/pose_opt.md) - Camera pose refinement
   - [Utils](./modules/utils.md) - Utility functions and helpers
   - [Random](./modules/random.md) - Random number generation

## Quick Start

```bash
# Build the project
./build.sh

# Train with a config file
./config_train -c configs/garden.json

# Train with visualization
./config_train -c configs/garden.json -v
```

## Dependencies

- CUDA 12.4+ (12.6-12.9 recommended)
- GCC 11+ (C++20 support)
- OpenCV
- spdlog, nlohmann_json, glm, OpenMP, NVTX3 (fetched via CMake)

## Features

- Pure C++/CUDA implementation with C++20 support
- Fused operations for efficient training and inference
- FP16 support for faster training (WIP)
- Adam optimizer with optional AdamW mode (`decouple_decay=true`)
- Multiple densification strategies (Default, Improved, MCMC)
- Multiple rasterizer backends (Default 3DGS, FastGS)
