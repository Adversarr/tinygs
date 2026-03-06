# Multiplatform Roadmap

## Purpose
Define a long-term, phase-based plan to evolve tinygs from CUDA-centric internals to a clean multiplatform runtime architecture while keeping training quality and maintainability stable.

This roadmap is the refactoring guide for future iterations.

## Scope
- In scope: CUDA, HIP, Metal.
- In scope: backend/device/buffer/event/queue abstraction.
- In scope: unified error/status model via `BackendError`.
- In scope: memory model standardization (allocation, free, copy, alignment, accessibility, interop).
- Out of scope: Vulkan.
- Out of scope: graph capture/replay as a core requirement.

## Long-Term Outcome
- One backend-agnostic runtime contract for tensor operations and memory movement.
- Backend-specific behavior isolated behind explicit backend modules.
- Library architecture supports both CLI training and future GUI integration without redesign.
- New backends can be added with bounded effort and clear acceptance gates.

## Guiding Principles
- Preserve correctness and CUDA behavior while refactoring.
- Prefer explicit contracts over implicit backend assumptions.
- Keep data ownership, memory accessibility, and synchronization explicit.
- Enforce strict layering: backend-agnostic layers must not depend on vendor runtimes.
- Use fail-fast behavior for unsupported backends, invalid configs, and capability mismatches.
- Ship in phases with measurable exit criteria.

## Architectural Constraints
- Single backend alive per process.
- Single device per process.
- No cross-backend memory transfer or interop required.
- Backend selection at compile time via CMake.
- Kernels and algorithms implemented per-backend.
- Global device context; no multi-device orchestration.

## Target Platform Model
### Core Runtime Entities
- Backend: runtime implementation boundary (CUDA, HIP, Metal).
- Device: selected compute target within a backend.
- Queue: execution queue abstraction (stream/command queue semantics).
- Buffer: memory object with explicit location, accessibility, and alignment contract.
- Event: synchronization primitive for queue ordering and host wait/query.
- Capability Profile: backend feature declaration used for validation and fallback.
- `BackendError`: unified operation status/result domain.

### BackendError Contract
`BackendError` is a required status type for all backend runtime operations, analogous in role to vendor error codes.

Requirements:
- Every runtime operation returns a `BackendError` status.
- Success and failure states are standardized across backends.
- Backend-native failures are mapped into stable cross-backend categories.
- Error reporting retains backend identity and operation context.
- Unknown/unsupported behavior is represented explicitly, never silently ignored.

Minimum error categories:
- Success
- Invalid Argument
- Unsupported
- Out Of Memory
- Timeout
- Device Lost
- Synchronization Error
- Runtime Failure
- Unknown Failure

## Memory Model Goals
- All allocations are described by explicit intent: memory class, access scope, and alignment.
- Copy semantics are explicit by source and destination memory class.
- Host-visible and device-local usage is explicit, not inferred.
- Interop-capable buffers are first-class for GUI and external runtime integration.
- Alignment requirements are defined centrally and enforced consistently.

## Queue and Synchronization Model
- Queue abstraction models execution order and async dispatch semantics.
- Event abstraction models queue-to-queue and host synchronization semantics.
- Queue/event APIs are backend-agnostic at the contract level.
- Command flow is queue-based; this roadmap does not require graph replay.

## Current Status Baseline (Updated 2026-03-05)

### Phase Status Summary
| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Governance and Guardrails | ✅ COMPLETE | Boundary checks enforced |
| Phase 1: Runtime Contract Foundation | ✅ COMPLETE | Full CUDA implementation |
| Phase 2: Memory/Error Contract | ✅ COMPLETE | Tests passing |
| Phase 3: Public API Decoupling | ✅ COMPLETE | All public headers backend-neutral |
| Phase 4: CUDA Path Migration | ⏳ NOT STARTED | Algorithm migration deferred |
| Phase 5: HIP Bring-Up | ⏳ NOT STARTED | Placeholder only |
| Phase 6: Metal Bring-Up | ⏳ NOT STARTED | Placeholder only |
| Phase 7: GUI/Interop Readiness | ⏳ NOT STARTED | Blocked on Phase 5-6 |
| Phase 8: Hardening | ⏳ NOT STARTED | Blocked on Phase 5-6 |

### Detailed Implementation Status

#### Runtime Contract (Phases 0-2) ✅
**Location:** `tinygs/include/tinygs/platform/`

- `backend_types.hpp`: `BackendType`, `BackendConfig`, `BackendStream`
- `backend_error.hpp`: `BackendError`, `BackendErrorCode` (9 categories)
- `backend_build.hpp`: Compile-time selection via `TINYGS_BACKEND_{CUDA|HIP|METAL}`
- `runtime.hpp`: `BackendRuntime`, `BackendQueue`, `BackendEvent`, `BackendBuffer`, `CapabilityProfile`
- `runtime_factory.hpp`: `create_backend_runtime()`
- `buffer_utils.hpp`: Helpers (`create_device_buffer`, `copy_from_host`, `copy_to_host`, `fill_buffer_zero`, `clone_buffer`, `resize_buffer`)
- `buffer_view.hpp`: `BufferView` for buffer slices

**CUDA Implementation:** `tinygs/src/cuda/runtime_cuda.cpp` (694 lines)
- `CudaRuntime`, `CudaQueue`, `CudaEvent`, `CudaBuffer`
- Full queue/event/buffer lifecycle
- Copy operations with overflow-safe region validation
- Memory classes: `Device`, `Unified` (HostPinned planned)

**Tests:** `test/backend_runtime_contract_test.cpp`, `test/backend_types_test.cpp`

**Boundary Enforcement:** `scripts/check_backend_boundaries.py`
- Allowlist: `scripts/policy/backend_boundary_allowlist.txt`
- Tracks vendor includes/types in public headers

#### Module Integration (Phase 3) ✅
All major module constructors accept `std::shared_ptr<BackendRuntime>`:

| Module | Interface File | Uses BackendBuffer |
|--------|---------------|-------------------|
| Orchestrator | `orchestrator.hpp:133,257-258,266-268` | Yes (m_loss_buffer, m_render_buffer, m_image_grad_buffer) |
| DataLoader | `dataloader.hpp:53,102` | Yes (via InternalCudaBuffer) |
| Dataset | `dataset.hpp:39` | Yes |
| Rasterizer | `rasterizer.hpp:90,121,130` | Yes |
| Optimizer | `optim.hpp:99,167,186` | Yes (Adam uses BackendBuffer for moments) |
| Strategy | `strategy.hpp:52,114,131` | Yes |
| GPUGaussian3d | `core/gpu_gaussian.hpp:28` | Yes (m_means, m_opacities, etc.) |

**Backend-Agnostic Math Types:**
- `tinygs/math/vec.hpp`: GLM-based vec2/3/4, mat3x3/4x4, quat (no CUDA dependencies)
- `DeviceSpan<T>`: Non-owning device memory view (replaces thrust::device_vector in public interfaces)
- All public headers pass boundary check with zero violations

#### HIP/Metal Placeholders ⏳
`runtime_factory.cpp:24-36` returns `BackendErrorCode::Unsupported` with message "planned but not implemented yet."

### Phase 3 Completion Summary
Phase 3 is complete. All public headers are backend-neutral:
- `tinygs/math/vec.hpp` provides GLM-based math types (no CUDA dependencies)
- `DeviceSpan<T>` replaces `thrust::device_vector<T>` in public interfaces
- Boundary check passes with zero violations

### Phase 2 Status Tracking
See `docs/MULTIPLATFORM_PHASE2_STATUS.md` for gate checklist and validation snapshot.
See `docs/MULTIPLATFORM_PHASE2_MEMORY_AND_ERROR.md` for memory descriptor and copy region contracts.

## Roadmap Phases
### Phase 0: Governance and Guardrails ✅ COMPLETE
Goal:
- Make refactoring safe and auditable before wide interface migration.

Outputs:
- `scripts/check_backend_boundaries.py`: Enforces layering rules
- `scripts/policy/backend_boundary_allowlist.txt`: Baseline for no-regression
- Rules: public headers must not include vendor headers; platform layer must be vendor-free

Exit Criteria:
- [x] Team-aligned rules for allowed dependencies and failure handling.
- [x] CI/policy checks defined for boundary violations.

### Phase 1: Runtime Contract Foundation ✅ COMPLETE
Goal:
- Establish backend runtime contract as the single operation surface.

Outputs:
- `BackendRuntime`, `BackendQueue`, `BackendEvent`, `BackendBuffer` interfaces
- `CapabilityProfile` for feature queries
- `BackendError`/`BackendErrorCode` unified status
- `backend_build.hpp` for compile-time backend selection
- CUDA implementation: `tinygs/src/cuda/runtime_cuda.cpp`

Exit Criteria:
- [x] Contract is complete enough to express current CUDA execution behavior.
- [x] No unresolved contract ambiguity for ownership, ordering, or status reporting.

### Phase 2: BackendError and Memory Contract Standardization ✅ COMPLETE
Goal:
- Standardize status/error and memory semantics before backend expansion.

Outputs:
- `BufferDesc`: memory_class, host_access, interop_mode, alignment, debug_name
- `BufferMemoryClass`: Device, Unified, HostPinned
- `BufferHostAccess`: None, Read, Write, ReadWrite
- `CopyRegion`, `BufferTransferRegion` for explicit copies
- `buffer_utils.hpp` helper library

Exit Criteria:
- [x] Runtime operations have deterministic status behavior.
- [x] Memory behavior is spec-defined and backend-independent at the contract level.
- [x] Tests pass: `ctest -R "BackendTypesTest|BackendRuntime"`

### Phase 3: Public API Decoupling ✅ COMPLETE
Goal:
- Remove backend-specific types from public-facing interfaces.

Outputs:
- `tinygs/math/vec.hpp`: GLM-based backend-agnostic math types
- `DeviceSpan<T>`: Non-owning device memory view for public interfaces
- All module constructors accept `std::shared_ptr<BackendRuntime>`
- `GPUGaussian3d` uses `BackendBuffer` internally
- `Orchestrator` uses `BackendBuffer` for loss/render/grad buffers

Exit Criteria:
- [x] Public interface is backend-neutral and stable for future backends.
- [x] No vendor runtime assumptions leak into public contracts.
- [x] Boundary check passes with `--strict-public-zero` flag.

### Phase 4: CUDA Path Migration to New Runtime ⏳ NOT STARTED
Goal:
- Make CUDA implementation conform fully to the new runtime contract.

Status:
- Deferred until needed for HIP/Metal bring-up. Current CUDA implementation is stable and performant.
- Public interfaces are backend-neutral (Phase 3 complete), so implementation can remain CUDA-specific.

When Resumed:
- Migrate direct CUDA API calls to runtime methods
- Consider backend-agnostic algorithm primitives vs per-backend implementations
- Key files: `gpu_gaussian.cu`, `orchestrator.cu`, `strategy/*.cu`

Exit Criteria:
- [ ] CUDA remains functional and performance-stable within agreed budgets.
- [ ] Backend-agnostic layers (platform/, public interfaces) are free of direct vendor runtime calls.
- [ ] HIP backend can be implemented without algorithm changes.

### Phase 5: HIP Bring-Up ⏳ NOT STARTED
Goal:
- Enable functional HIP backend using the same runtime contract.

Prerequisites:
- Phase 4 complete (CUDA path migration)
- Backend-agnostic algorithm strategy decided

Outputs:
- `tinygs/src/hip/runtime_hip.cpp`: HIP implementation of `BackendRuntime`
- `tinygs/src/hip/` HIP-specific kernels
- HIP-enabled training slice for core forward/backward/optimization flow

Planned Approach:
1. Implement `HipRuntime`, `HipQueue`, `HipEvent`, `HipBuffer`
2. Map HIP errors to `BackendErrorCode`
3. Port CUDA kernels to HIP (or use hipify)
4. Implement algorithm primitives (sort, reduce) for HIP

Exit Criteria:
- [ ] HIP smoke training passes on reference scenes.
- [ ] Numerical parity stays within approved tolerance against CUDA baseline.

### Phase 6: Metal Bring-Up ⏳ NOT STARTED
Goal:
- Enable functional Metal backend with queue-based execution semantics.

Prerequisites:
- Phase 5 complete (HIP bring-up for reference)
- Metal shader compilation pipeline

Outputs:
- `tinygs/src/metal/runtime_metal.mm`: Metal implementation
- Metal shaders (MSL) for rasterization
- Metal execution path for core training/inference slice

Key Challenges:
- Metal uses command buffers/encoders, not streams
- Shader language translation (CUDA → MSL)
- No direct equivalent to CUDA graphs (if used)

Exit Criteria:
- [ ] Metal smoke runs pass on supported hardware.
- [ ] Contract-level behavior parity verified against CUDA baseline expectations.

### Phase 7: GUI and Interop Readiness ⏳ NOT STARTED
Goal:
- Ensure architecture supports external rendering/UI integration cleanly.

Prerequisites:
- Phase 4 complete
- `BufferInteropMode::External` implemented

Outputs:
- Interop-capable buffer/event model validated
- Forward/backward invocation model suitable for library embedding
- Examples for GUI integration (OpenGL/Vulkan interop)

Exit Criteria:
- [ ] External integration path avoids ad-hoc backend-specific glue.
- [ ] Ownership and synchronization contracts are sufficient for GUI workflows.

### Phase 8: Parity, Hardening, and Lifecycle Maintenance ⏳ NOT STARTED
Goal:
- Stabilize multiplatform development with enforceable quality gates.

Prerequisites:
- Phases 5-6 complete

Outputs:
- Cross-backend parity test strategy and tolerance policy
- Performance baseline tracking and regression gates
- Capability reporting and fallback/error messaging policy

Exit Criteria:
- [ ] CI matrix and release process include multiplatform quality gates.
- [ ] Backends meet functional, numerical, and stability acceptance criteria.

## Cross-Phase Acceptance Gates
Each phase must satisfy:
- Functional correctness on targeted scope.
- Deterministic `BackendError` behavior.
- Clear ownership and synchronization semantics.
- No layering regressions.
- Documented risk review and rollback strategy.

## Testing Strategy (Roadmap-Level)
- Contract tests: runtime entity lifecycle, memory semantics, queue/event ordering, status behavior.
- Integration tests: end-to-end training slice per backend.
- Parity tests: metric and loss tolerance comparisons across backends.
- Regression tests: performance and memory stability trend checks.

## Risk Register
- Risk: abstraction drift causes hidden backend-specific behavior.
  Mitigation: strict contract tests and boundary enforcement.

- Risk: memory semantics diverge by backend and break correctness.
  Mitigation: explicit memory/accessibility/alignment policy with mandatory validation.

- Risk: parity regressions during backend bring-up.
  Mitigation: staged parity gates and fixed-scene baselines.

- Risk: short-term refactor velocity drops.
  Mitigation: narrow phase scope, clear exit criteria, and incremental stabilization.

## Default Decisions for This Roadmap
- Planned API break is accepted for backend-neutral public interfaces.
- Backend rollout order: CUDA abstraction completion, then HIP, then Metal.
- Queue abstraction models stream/command-queue semantics.
- Graph replay/capture is not a required part of this roadmap.
- Backend selection at compile time via CMake (`-DTINYGS_BACKEND=CUDA/HIP/METAL`).
- Per-backend kernel files and algorithm implementations; no shared kernel source translation.
- Custom device algorithms per backend; no cross-platform algorithm library dependency.
- Global device pointer pattern; device context accessible throughout runtime.
- Rasterizer implementations are per-backend with shared algorithmic structure.

## Completion Definition
The roadmap is complete when:
- Runtime contract is the only cross-backend execution interface.
- `BackendError` is the mandatory status model for runtime operations.
- CUDA, HIP, and Metal satisfy agreed functional and parity gates.
- Library interfaces are stable, maintainable, and suitable for both CLI and GUI integration.
