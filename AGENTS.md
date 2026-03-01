# AGENTS.md

## Purpose

Develop tinygs, a C++/CUDA 3D Gaussian Splatting library. Focus on CUDA kernels, training pipelines, rasterization. Do not modify `cmake/` dependencies.

## Commands

```bash
./build.sh                                          # Build default targets
BUILD_TYPE=Debug ./build.sh                         # Debug build
TINYGS_CUDA_ARCHITECTURES="86" ./build.sh           # Build for RTX 30+ cards
TARGETS="config_train" ./build.sh                   # Build specific target
./config_train -c configs/garden.json               # Train
./config_train -c configs/garden.json -v -d         # Train with viz/debug
```

## Tech Stack

C++20, CUDA 12.4+, CMake 3.28+, GCC 11+, nvcc. Deps: OpenCV, spdlog, nlohmann_json, glm, OpenMP, NVTX3, cxxopts, Thrust.

## Repo Map

| Path | Purpose |
|------|---------|
| `tinygs/include/tinygs/` | Public headers |
| `tinygs/src/core/` | Gaussian primitives, camera models |
| `tinygs/src/cuda/` | CUDA utilities, memory management |
| `tinygs/src/rasterizer/` | Forward/backward rasterization kernels |
| `tinygs/src/optim/` | Optimizers (adamw, sgd, lion, adan) |
| `tinygs/src/loss/` | Loss functions (l1, ssim) |
| `tinygs/src/strategy/` | Densification strategies |
| `tinygs/src/dataloader/` | Data loading |
| `tinygs/src/initialization/` | Gaussian initialization |
| `examples/` | Executables (`config_train`, `video_to_png`, `single_gs`) |
| `configs/` | JSON training configurations |

## Code Style

- Functions/variables: `snake_case`; types: `CamelCase`; private members: `m_` prefix
- 2-space indent, 120 columns, `#pragma once`, braces on same line
- Import order: std → CUDA → third-party → `"tinygs/..."` (quoted with path)
- CUDA: use `__restrict__` on pointer args, `TINYGS_PRAGMA_UNROLL` for loops
- Kernels: bounds check with `if (idx >= n) return;`, use `n_blocks_linear()`/`N_THREADS_LINEAR`
- Errors: use `log_*` macros, `CUDA_CHECK_THROW()`, `CHECK_THROW()`
- JSON: check `config.contains()` before accessing optional fields

## Boundaries

**Always:** Run `./build.sh` after code changes. Pass `cudaStream_t` explicitly. Use `log_*` macros. Use `__restrict__` on kernel pointers. Check `config.contains()` for optional JSON fields. Use `n_blocks_linear()`/`N_THREADS_LINEAR` for kernel launches.

**Ask First:** Add dependencies. Modify CMake/CUDA arch defaults. Add rasterizers. Change training defaults. Modify `.clang-*` files.

**Never:** Modify `cmake/CPM.cmake` or fetched deps. Commit `build/`, `*.ply`, binaries. Use `std::cout`/`printf`. Skip bounds checks in kernels. Access optional JSON fields without `contains()`.
