# Multiplatform Phase 0 Guardrails

## Purpose
Phase 0 establishes a no-regression safety boundary while backend-neutral runtime contracts are introduced.

## Guardrail Rules
1. Public headers under `tinygs/include/tinygs/**` (except `tinygs/include/tinygs/cuda/**`) must not add new vendor includes or vendor types.
2. Platform-layer files under `tinygs/include/tinygs/platform/**` and `tinygs/src/platform/**` must not add new direct vendor dependencies.
3. Violations are enforced as no-regression against `scripts/policy/backend_boundary_allowlist.txt`.

## Enforcement
- Script: `scripts/check_backend_boundaries.py`
- Policy baseline: `scripts/policy/backend_boundary_allowlist.txt`
- Test wiring: `tinygs_phase0_backend_boundaries` in `test/CMakeLists.txt`

## Fail-Fast Expectations
1. Backend mismatch between config and compile-time backend must throw immediately.
2. Unsupported backends (HIP, METAL) remain explicit placeholders and must throw clearly.

## Expected Usage
```bash
python3 scripts/check_backend_boundaries.py check
python3 scripts/check_backend_boundaries.py scan
# Optional strict mode (Phase 3 readiness):
python3 scripts/check_backend_boundaries.py check --strict-public-zero
```
