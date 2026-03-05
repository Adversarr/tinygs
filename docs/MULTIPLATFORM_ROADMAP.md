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

## Current Status Baseline (2026-03-05)
- Backend configuration and backend type selection exist.
- CUDA is the only executable backend path.
- HIP and Metal are fail-fast placeholders.
- Public/runtime surfaces still carry substantial CUDA assumptions.
- Memory and temporary storage systems remain CUDA-specific.
- Existing tests validate backend type parsing/factory fail-fast behavior, not multiplatform runtime execution.

## Roadmap Phases
### Phase 0: Governance and Guardrails
Goal:
- Make refactoring safe and auditable before wide interface migration.

Outputs:
- Layering rules documented and enforced.
- Definition of backend-agnostic vs backend-specific boundaries.
- Refactor acceptance criteria and phase gates formalized.

Exit Criteria:
- Team-aligned rules for allowed dependencies and failure handling.
- CI/policy checks defined for boundary violations.

### Phase 1: Runtime Contract Foundation
Goal:
- Establish backend runtime contract as the single operation surface.

Outputs:
- Stable runtime contract for queue/event/buffer/device operations.
- Capability profile model for backend feature checks.
- Lifecycle rules for runtime objects.

Exit Criteria:
- Contract is complete enough to express current CUDA execution behavior.
- No unresolved contract ambiguity for ownership, ordering, or status reporting.

### Phase 2: BackendError and Memory Contract Standardization
Goal:
- Standardize status/error and memory semantics before backend expansion.

Outputs:
- `BackendError` adopted as mandatory operation status.
- Unified memory descriptors covering alignment, accessibility, and interop intent.
- Copy and synchronization semantics aligned across runtime operations.

Exit Criteria:
- Runtime operations have deterministic status behavior.
- Memory behavior is spec-defined and backend-independent at the contract level.

### Phase 3: Public API Decoupling
Goal:
- Remove backend-specific types from public-facing interfaces.

Outputs:
- Public API uses backend-agnostic runtime entities.
- Domain APIs no longer expose CUDA-native containers or handles.
- Planned API break communicated and versioned.

Exit Criteria:
- Public interface is backend-neutral and stable for future backends.
- No vendor runtime assumptions leak into public contracts.

### Phase 4: CUDA Path Migration to New Runtime
Goal:
- Make CUDA implementation conform fully to the new runtime contract.

Outputs:
- CUDA backend provides full runtime behavior through the new abstraction.
- Backend-agnostic modules consume runtime contract only.
- Existing CUDA training behavior remains intact.

Exit Criteria:
- CUDA remains functional and performance-stable within agreed budgets.
- Backend-agnostic layers are free of direct vendor runtime calls.

### Phase 5: HIP Bring-Up
Goal:
- Enable functional HIP backend using the same runtime contract.

Outputs:
- HIP runtime implementation with queue/event/buffer parity.
- HIP-enabled training slice for core forward/backward/optimization flow.

Exit Criteria:
- HIP smoke training passes on reference scenes.
- Numerical parity stays within approved tolerance against CUDA baseline.

### Phase 6: Metal Bring-Up
Goal:
- Enable functional Metal backend with queue-based execution semantics.

Outputs:
- Metal runtime implementation aligned with contract.
- Metal execution path for core training/inference slice.

Exit Criteria:
- Metal smoke runs pass on supported hardware.
- Contract-level behavior parity verified against CUDA baseline expectations.

### Phase 7: GUI and Interop Readiness
Goal:
- Ensure architecture supports external rendering/UI integration cleanly.

Outputs:
- Interop-capable buffer/event model validated.
- Forward/backward invocation model suitable for library embedding.

Exit Criteria:
- External integration path avoids ad-hoc backend-specific glue.
- Ownership and synchronization contracts are sufficient for GUI workflows.

### Phase 8: Parity, Hardening, and Lifecycle Maintenance
Goal:
- Stabilize multiplatform development with enforceable quality gates.

Outputs:
- Cross-backend parity test strategy and tolerance policy.
- Performance baseline tracking and regression gates.
- Capability reporting and fallback/error messaging policy.

Exit Criteria:
- CI matrix and release process include multiplatform quality gates.
- Backends meet functional, numerical, and stability acceptance criteria.

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
