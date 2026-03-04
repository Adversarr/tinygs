# CUDA Header Cleanup Plan

**Goal:** Remove all CUDA and thrust dependencies from public headers to enable cross-platform API stability.

---

## Current Blockers

| Header | Issue |
|--------|-------|
| `common.hpp` | `cuda_fp16.h`, `cuda_bf16.h` |
| `cuda/gpu_memory.hpp` | `cuda.h`, `cudaStream_t` |
| `cuda/common_host.hpp` | `cuda_runtime.h`, `cudaStream_t` in kernel launch helpers |
| `cuda/cuda_graph.hpp` | `cuda.h`, `cuda_runtime.h`, `cudaStream_t` |
| `cuda/multi_stream.hpp` | Heavy `cudaStream_t`/`cudaEvent_t` usage |
| `core/gpu_gaussian.hpp` | `thrust::device_vector` in storage and return types |
| `optim/adam.hpp` | `thrust::device_vector` in moment buffers |
| `optim/adam_per_gaussian.hpp` | `thrust::device_vector` in step counts and moments |
| `optim/optim.hpp` | `cudaStream_t` in `step()` interface |
| `strategy/fastgs.hpp` | `thrust::device_vector` for importance/pruning scores |
| `rasterizer/rasterizer.hpp` | `cudaStream_t` parameter |
| `loss/loss.hpp` | `cudaStream_t` member |
| `dataloader/dataloader.hpp` | `cudaStream_t` in `transfer_gpu()` |
| `dataloader/simple.hpp` | `cudaStream_t` in `next()` |
| `random/multinomial.hpp` | `cudaStream_t` parameters |
| `utils/image_format.hpp` | `cudaStream_t` in GPU conversion functions |
| `orchestrator.hpp` | Transitively includes all above |

---

## Phase 1: Foundation Types

Create `tinygs/core/types.hpp` with opaque handles (`CudaStream = void*`, `DevicePtr = void*`).

Create `tinygs/core/glm_types.hpp` with GLM typedefs only - no CUDA macros.

**Verify:** These files must compile without CUDA toolkit installed.

---

## Phase 2: GPUGaussianBase Interface

Create `tinygs/core/gpu_gaussian_base.hpp` - pure abstract class with raw pointer accessors (`vec3* means_data()`, not `thrust::device_vector<vec3>& means()`).

Update `gpu_gaussian.hpp` to inherit from base and implement raw pointer accessors via `thrust::raw_pointer_cast()`. Keep deprecated thrust accessors for backward compatibility.

Update `gpu_gaussian.cu` to implement the new interface methods.

---

## Phase 3: Context Objects

Create `tinygs/rasterizer/rasterizer_types.hpp` with `RasterizeContextBase` containing only platform-independent fields (`CudaStream stream`, not `cudaStream_t`).

Update `rasterizer.hpp` to use base context and accept `GPUGaussianBase*` instead of `GPUGaussian3d*`.

---

## Phase 4: Module Interfaces

Create base interfaces for each module:
- `tinygs/optim/optim_base.hpp` → `OptimizerBaseInterface`
- `tinygs/strategy/strategy_base.hpp` → `StrategyBaseInterface`
- `tinygs/dataloader/dataloader_base.hpp` → `DataLoaderBaseInterface`
- `tinygs/loss/loss_base.hpp` → `LossBaseInterface`

Each uses `CudaStream` and `GPUGaussianBase` instead of CUDA types.

Update existing headers to inherit from base interfaces.

---

## Phase 5: Orchestrator Header

Create `tinygs/orchestrator_types.hpp` with `OrchestratorConfig` and `TrainingState` (no CUDA fields).

Update `orchestrator.hpp` to:
- Use `GPUGaussianBase*` instead of `GPUGaussian3d*`
- Use `CudaStream` instead of `cudaStream_t`
- Move `GPUMemory<T>` buffers to internal `Impl` struct (Pimpl pattern)

Create `tinygs/src/orchestrator_impl.hpp` for CUDA-specific internals used by `.cu` file.

---

## Phase 6: Build System

Add `TINYGS_NO_CUDA` CMake option.

Add compile test that includes all `_base.hpp` and `types.hpp` files without CUDA toolkit.

Add CI workflow to verify header cleanliness.

---

## Phase 7: Migration

Document old API → new API mappings with deprecation timeline.

Add `[[deprecated]]` attributes to old API functions.

---

## Execution Order

Start with Phase 1 (no dependencies, immediately verifiable). Each subsequent phase depends on the previous. Phases 2 and 5 have highest risk - test thoroughly.
