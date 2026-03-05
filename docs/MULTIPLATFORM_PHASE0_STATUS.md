# Multiplatform Phase 0 Status

## Scope
- Date: 2026-03-05
- Owner: adversarr
- Branch/commit: `cleanup` / `040a7f3`

## Gate Checklist
- [x] Guardrail spec present and linked:
  - [x] `docs/MULTIPLATFORM_PHASE0_GUARDRAILS.md`
  - [x] linked from roadmap and docs index
- [x] Boundary checker exists:
  - [x] `scripts/check_backend_boundaries.py`
  - [x] `scripts/policy/backend_boundary_allowlist.txt`
- [x] No-regression enforcement wired in tests (`ctest`)
- [x] Compile-time backend selector added (`TINYGS_BACKEND`)
- [x] HIP/METAL configure-time fail-fast validated

## Boundary Check Snapshot
- Command: `python3 scripts/check_backend_boundaries.py check`
- Result: pass (`Boundary policy check passed: no new violations.`)
- New/increased violations: `0` expected
- Notes: also executed in `ctest` via `tinygs_phase0_backend_boundaries`.

## CMake Backend Gate Snapshot
- CUDA configure:
  - Command: `cmake -S . -B build/Release -DTINYGS_BUILD_TESTS=ON`
  - Result: pass (configured with `Backend: CUDA`)
- HIP configure (expected fail):
  - Command: `cmake -S . -B /tmp/tinygs_phase0_hip -DTINYGS_BACKEND=HIP -DTINYGS_BUILD_APPS=OFF -DTINYGS_BUILD_TESTS=OFF`
  - Result: expected fail
  - Error substring verified: `planned but not implemented yet`
- METAL configure (expected fail):
  - Command: `cmake -S . -B /tmp/tinygs_phase0_metal -DTINYGS_BACKEND=METAL -DTINYGS_BUILD_APPS=OFF -DTINYGS_BUILD_TESTS=OFF`
  - Result: expected fail
  - Error substring verified: `planned but not implemented yet`

## Known Baseline Counts
- `public_header_forbidden_include`: `21`
- `public_header_forbidden_vendor_type`: `66`
- `platform_layer_forbidden_vendor_dep`: `2`

## Risks / Follow-ups for Phase 1
- Convert no-regression baseline into stricter scope as CUDA symbols are removed from public headers.
- Add CI provider wiring (GitHub Actions or equivalent) to run boundary checks on PRs.
- Expand platform abstraction tests once runtime contract interfaces are introduced.

## Rollback Notes
- If unexpected breakage occurs, revert policy wiring first, keep docs and rationale, then reintroduce with narrowed scope.
