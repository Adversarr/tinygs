# Multi-Platform Roadmap (CUDA + HIP + Metal)

## Scope
- In scope: CUDA, HIP, Metal.
- Out of scope for now: Vulkan (explicitly deferred due implementation complexity and maintenance overhead).
- Objective: keep one trainer architecture and one public API, with backend-specific execution hidden behind platform seams.

## Guiding Principles
- Preserve current CUDA performance and behavior while introducing abstraction incrementally.
- Isolate backend-specific types from public interfaces first (`BackendType`, `BackendStream`, backend factory/context).
- Reuse shared algorithmic structure across CUDA/HIP/Metal where practical (loss orchestration, optimizer semantics, strategy flow, scheduling).
- Keep fail-fast behavior for unsupported backends and unknown config values.

## Current Codebase Status (Audit)
Date: 2026-03-04

### Architecture and coupling
- Training pipeline is orchestrator-centric and stable (`dataloader -> rasterizer -> loss -> optimizer -> strategy`).
- CUDA stream usage is pervasive across runtime-critical modules.
- Factories exist for major components and already provide an extension seam.
- Rasterizer/optimizer internals are CUDA-first and assume CUDA kernel launch + CUDA memory utilities.

### Abstraction baseline
- Added platform abstraction layer:
  - `tinygs/include/tinygs/platform/backend_types.hpp`
  - `tinygs/include/tinygs/platform/backend_context.hpp`
  - `tinygs/include/tinygs/platform/backend_factory.hpp`
  - `tinygs/src/platform/backend_factory.cpp`
- `BackendType` currently supports: `cuda`, `hip`, `metal`.
- Runtime behavior today:
  - CUDA backend context is constructible.
  - HIP/Metal fail fast with explicit "planned but not implemented yet" errors.

### API migration status
- Public interfaces now use `BackendStream` in critical orchestration surfaces:
  - Gaussian ops (`clone_async`, `memset_async`, `reorder`)
  - Dataloader transfer/next methods
  - Loss/Rasterizer context stream fields
  - Optimizer step interfaces
  - Random/image utility stream parameters
  - Orchestrator major stream member

### Configuration/runtime status
- `apps/config_train` now reads backend config, creates backend context, and applies CUDA device selection.
- Default/sample configs updated with explicit backend section (`cuda`, device `0`).
- Backend setting is currently validated to CUDA-only runtime execution.

### Testing status
- Test coverage has been strengthened this iteration:
  - Re-enabled LR scheduler unit test target.
  - Added backend type/factory tests for parsing, roundtrip, and fail-fast behavior.
- Remaining gap: no HIP/Metal runtime test path yet (blocked by backend implementation).

## Progressive Implementation Plan

## Phase 0: Foundation and Safety Rails
- Deliverables:
  - Backend enums/types/config parsing and factory/context layer.
  - CUDA-only execution remains default and fully functional.
  - Explicit errors for HIP/Metal selection.
- Exit criteria:
  - All existing CUDA tests pass.
  - Configured backend selection works for CUDA and fails clearly for others.

## Phase 1: Stream and Launch Boundary Cleanup
- Deliverables:
  - Backend stream type used at module boundaries.
  - Central conversion helpers at CUDA boundary (`to_cuda_stream`, `to_backend_stream`).
  - No behavior change in CUDA code path.
- Exit criteria:
  - Build succeeds without regressions.
  - Smoke training run succeeds with CUDA backend.

## Phase 2: Backend Runtime Interface
- Deliverables:
  - Introduce backend runtime operations interface (stream create/destroy/sync, memcpy, memset, event/scratch ops).
  - Replace direct orchestrator/runtime CUDA calls with backend runtime wrappers.
- Exit criteria:
  - CUDA backend uses runtime interface end-to-end.
  - No direct CUDA runtime calls remain in backend-agnostic orchestration layer.

## Phase 3: HIP Backend Bring-up
- Deliverables:
  - HIP backend context + runtime implementation.
  - HIP build target and compile-time guards.
  - Shared kernels/utilities ported where practical; HIP-specific implementations where required.
- Exit criteria:
  - Functional HIP smoke training on a reference config.
  - Core tests enabled under HIP build variant.

## Phase 4: Metal Backend Bring-up
- Deliverables:
  - Metal backend context + runtime operations.
  - Metal rasterization/loss/optimizer dispatch strategy (likely split implementation files).
  - Apple build path and CI lane.
- Exit criteria:
  - Functional Metal smoke training on a reference config.
  - Core regression tests enabled under Metal-capable build.

## Phase 5: Cross-Backend Parity and Hardening
- Deliverables:
  - Numerical parity budgets and tolerance checks across CUDA/HIP/Metal.
  - Backend capability reporting and graceful fallback messages.
  - Performance baseline tracking and regression gates.
- Exit criteria:
  - Defined parity thresholds met for representative scenes.
  - CI matrix includes CUDA + HIP + Metal checks (where runners available).

## Testing Strengthening Plan
- Unit tests:
  - Backend config parsing and factory behavior.
  - Scheduler and optimizer API contract tests (already improved this iteration).
- Integration tests:
  - End-to-end config wiring with backend selection.
  - Factory composition checks in `config_train` path.
- Smoke tests:
  - Short training run on canonical config for CUDA each PR.
  - Add HIP/Metal smoke lanes once each backend is executable.
- Regression checks:
  - Loss/metric sanity ranges.
  - Basic throughput sanity to catch severe perf regressions.

## Progress Tracker

## Iteration: 2026-03-04
- [x] Replaced draft roadmap with phased CUDA/HIP/Metal plan (Vulkan deferred).
- [x] Added backend abstraction headers and backend factory source.
- [x] Migrated key public module stream interfaces to `BackendStream`.
- [x] Added stream conversion helpers and compatibility launch wrappers.
- [x] Wired backend parsing + CUDA device selection in `config_train`.
- [x] Updated sample configs with explicit backend section.
- [x] Re-enabled `tinygs_lr_scheduler_test` target.
- [x] Added `tinygs_backend_types_test` target and tests.
- [ ] Introduce backend runtime operation interface (Phase 2).
- [ ] HIP runtime + kernel bring-up (Phase 3).
- [ ] Metal runtime + kernel bring-up (Phase 4).

## Risks and Mitigations
- Risk: stream/memory APIs stay CUDA-leaky in orchestrator and strategy internals.
  - Mitigation: Phase 2 runtime interface with strict layering rules.
- Risk: parity divergence across backends in optimizer/loss numerics.
  - Mitigation: add parity tests with explicit tolerances before enabling default backend switching.
- Risk: test coverage lags behind backend expansion.
  - Mitigation: every phase includes required unit + smoke gates before phase completion.
