# Project Guidelines

## Code Style
- Naming/formatting: `snake_case` functions/vars, `CamelCase` types, `m_` private members, 2-space indent, same-line braces.
- Include order: `std -> CUDA -> third-party -> "tinygs/..."` (see `tinygs/src/orchestrator.cu`, `tinygs/src/strategy/strategy.cpp`).
- Logging/errors: use `log_*`, `CUDA_CHECK_THROW()`, `CHECK_THROW()`; avoid new `std::cout`/`printf` in runtime/library code.
- CUDA kernels: `__restrict__` pointer args, first-line bounds checks, launch via `n_blocks_linear()`/`N_THREADS_LINEAR` (see `tinygs/src/loss/l1.cu`).
- Optional JSON fields must use `config.contains(...)` guards (see `apps/config_train.cpp`, `tinygs/src/optim/optim.cu`).

## Architecture
- Orchestrator-centered pipeline: dataloader -> rasterizer -> loss/metric -> optimizer -> strategy (`tinygs/src/orchestrator.cu`, `docs/ARCHITECTURE.md`).
- Public interfaces live in `tinygs/include/tinygs/**`; implementations in `tinygs/src/**`; app entrypoints are in `apps/`.
- Factory seams: `create_dataset`, `create_dataloader`, `create_rasterizer`, `create_optimizer`, `create_strategy`, `create_loss`.
- Current rasterizer types: `default`, `fastgs` (`tinygs/src/rasterizer/rasterizer.cpp`).
- Current optimizer types: `adam`, `adam_per_gaussian` / `adam_pg` (`tinygs/src/optim/optim.cu`).

## Repo Map
- `apps/`: CLI entrypoints (`config_train`, `export_default`, `single_gs`) and JSON wiring in `apps/config_train.cpp`.
- `tinygs/include/tinygs/`: public module interfaces (core, rasterizer, optim, strategy, loss, dataloader, dataset).
- `tinygs/src/orchestrator.cu`: training loop orchestration and component coordination.
- `tinygs/src/rasterizer/`, `tinygs/src/optim/`, `tinygs/src/strategy/`, `tinygs/src/loss/`: primary training modules.
- `tinygs/src/dataloader/`, `tinygs/src/dataset/`, `tinygs/src/initialization/`: data ingestion and initialization path.
- `configs/`: runnable JSON configs (e.g., `configs/garden.json`).
- `test/`: C++ tests (`test/basic_test.cpp`) and test target wiring (`test/CMakeLists.txt`).
- `cmake/` + root `CMakeLists.txt`: dependency and build integration points.

## Build and Test
```bash
./build.sh
BUILD_TYPE=Debug ./build.sh
TARGETS="config_train export_default single_gs" ./build.sh
TINYGS_CUDA_ARCHITECTURES="86" ./build.sh
cmake -S . -B build/Release -DTINYGS_BUILD_TESTS=ON
cmake --build build/Release --target tinygs_basic_test -j "$(nproc)"
ctest --test-dir build/Release --output-on-failure
./config_train -c configs/garden.json
```
- After C++/CUDA changes: run `./build.sh`; run tests when affected; smoke-test training-path changes.
- `docs/CLI.md` may drift; treat `build.sh` and `apps/CMakeLists.txt` as source of truth for targets.

## Project Conventions
- Keep patches surgical and module-local; if adding files, update explicit CMake target lists.
- Pass `cudaStream_t` explicitly through call chains; do not add hidden/default-stream behavior.
- Config parsing convention: required fields via `.at(...)`, optional fields via `contains(...)`.

## Integration Points
- Dependency/linkage integration is centralized in `cmake/dependencies.cmake`, root `CMakeLists.txt`, and `tinygs/CMakeLists.txt`.
- JSON-driven runtime composition is centralized in `apps/config_train.cpp`; wire new modules there.
- Treat `ref_impl/` as reference-only; do not derive production style/policy from it.

## Security
- Keep fail-fast behavior for unknown component/type strings in factories.
- Be careful with config-provided paths (`root_path`, output dirs); avoid adding implicit path assumptions.

## Boundaries
- **Always:** minimal focused changes; preserve kernel safety checks; use project logging/error macros.
- **Ask first:** adding dependencies, changing CMake defaults/CUDA arch defaults, adding rasterizer implementations, changing default training configs/behavior, modifying `.clang-*`.
- **Never:** modify `cmake/CPM.cmake` or fetched deps, commit generated artifacts (`build/`, binaries, `*.ply`, `*.pt`), remove kernel bounds checks.
