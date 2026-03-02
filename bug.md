# CUDA Illegal Memory Access Bug Report

## Executive Summary

Comprehensive review of all strategy files (`improved.cu`, `default.cu`, `fastgs.cu`, `absgs.cu`, `mcmc.cu`) and `orchestrator.cu` identified **multiple critical and moderate bugs**.

### Key Findings

1. **Use-After-Free: NOT PRESENT** - All strategy files correctly call `StrategyBase::on_duplicate()` BEFORE capturing raw pointers. This was verified across all files.

2. **Stream Issues: CRITICAL** - `mcmc.cu` has **15+ instances** of using `thrust::device` (default stream) instead of `ctx.stream`, causing potential race conditions. **ADDITIONAL**: `orchestrator.cu:mean()` also uses `thrust::device`.

3. **Synchronous Memset: WIDESPREAD** - All strategy files use synchronous `memset(0)` instead of async `memset_async(ctx.stream, 0)`.

4. **Missing Execution Policies** - Several thrust operations in `mcmc.cu`, `absgs.cu`, and `default.cu` lack explicit execution policies, defaulting to the default stream.

5. **Additional Stream Issues** - Other files (`image_format.cu`, `psnr.cu`) also use `nullptr` for stream parameter in `linear_kernel` calls.

---

## Bug 1: CRITICAL - Stream Mismatch in `orchestrator.cu:reorder_gaussians()` AND `mean()`

**File:** `tinygs/src/orchestrator.cu`  
**Lines:** 232-241 (`mean()`), 1118-1140 (`reorder_gaussians()`)

### Description

Two separate stream issues exist in this file:

1. The `mean()` helper function uses `thrust::device` (default stream) instead of accepting a stream parameter.
2. The `reorder_gaussians()` function uses the default stream (nullptr) for some operations while the main training pipeline uses `m_major_stream`.

### Root Cause

```cpp
// Lines 232-241 - mean() uses thrust::device!
void mean(const vec3* data, size_t size, vec3& out) {
  out = thrust::transform_reduce(
    thrust::device,  // WRONG - should accept stream parameter
    data,
    data + size,
    ...
  );
}

// Line 1125 - reorder_gaussians() passes nullptr to reorder()
auto idx = reorder(thrust::raw_pointer_cast(pos.data()), n, nullptr);

// Lines 1128-1130 - m_gaussians/gradients/optimizer reorder don't pass stream
m_gaussians->reorder(thrust::raw_pointer_cast(idx.data()));  // Missing stream
m_gradients->reorder(thrust::raw_pointer_cast(idx.data()));  // Missing stream  
m_optimizer->reorder(thrust::raw_pointer_cast(idx.data()));  // Missing stream

// Line 1134 - linear_kernel uses nullptr stream
linear_kernel(densification_update, 0, nullptr, n, ...);
```

### Impact

- **Race condition**: Reading `densification_info` before forward/backward completes
- **Non-deterministic results**: Inconsistent densification state
- **Scene scale recomputation**: The `mean()` function in `recompute_scene_scale()` uses default stream

### Recommended Fix

For `mean()` function, add stream parameter:
```cpp
void mean(const vec3* data, size_t size, vec3& out, cudaStream_t stream) {
  out = thrust::transform_reduce(
    thrust::cuda::par.on(stream),  // Use provided stream
    ...
  );
}
```

For `reorder_gaussians()`, pass `m_major_stream`:
```cpp
void Orchestrator::reorder_gaussians() {
  auto idx = reorder(thrust::raw_pointer_cast(pos.data()), n, m_major_stream);
  
  m_gaussians->reorder(thrust::raw_pointer_cast(idx.data()), m_major_stream);
  m_gradients->reorder(thrust::raw_pointer_cast(idx.data()), m_major_stream);
  m_optimizer->reorder(thrust::raw_pointer_cast(idx.data()), m_major_stream);

  if (m_rasterize_ctx.densification_info) {
    auto new_info = std::make_shared<GPUBuffer<tinygs::DensificationInfo>>(n);
    linear_kernel(densification_update, 0, m_major_stream, n,
                  m_rasterize_ctx.densification_info->data(),
                  new_info->data(),
                  thrust::raw_pointer_cast(idx.data()));
    m_rasterize_ctx.densification_info = new_info;
  }
}
```

---

## Bug 2: CRITICAL - Massive Stream Mismatch in `mcmc.cu`

**File:** `tinygs/src/strategy/mcmc.cu`  
**Lines:** 221-538 (15+ instances)

### Description

Almost ALL thrust operations in `mcmc.cu` use `thrust::device` (default stream) instead of `ctx.stream`. This causes all operations to run without synchronization with the training stream.

### Root Cause

```cpp
// Lines 221-227 - uses thrust::device (default stream)
thrust::transform(
  thrust::device,  // WRONG - should use ctx.stream
  m_gaussians->opacities().begin(),
  m_gaussians->opacities().end(),
  opacities.begin(),
  [] __device__ (float opacity) { return activate_opacity(opacity); }
);

// Lines 240-245, 251-268, 274-277, 278-285, 301-304, 313-345, 356-362, 
// 366-388, 400-443, 448-492, 508-538 - ALL use thrust::device or missing exec
```

### Affected Lines

| Lines | Operation | Issue |
|-------|-----------|-------|
| 221-227 | `thrust::transform` | Uses `thrust::device` |
| 240-245 | `thrust::copy` | Uses `thrust::device` |
| 251-259 | `thrust::transform` | Missing `exec` |
| 260-268 | `thrust::transform` | Missing `exec` |
| 274-277 | `thrust::for_each` | Uses `thrust::device` |
| 278-285 | `thrust::transform` | Missing `exec` |
| 301-304 | `thrust::copy` | Missing `exec` |
| 313-345 | `thrust::for_each` | Missing `exec` |
| 356-362 | `thrust::transform` | Uses `thrust::device` |
| 366-388 | `thrust::transform` | Missing `exec` |
| 400-414 | `thrust::copy_if` | Missing `exec` |
| 418-427 | `thrust::transform` | Uses `thrust::device` |
| 435-443 | `thrust::transform` | Uses `thrust::device` |
| 448-465 | `thrust::transform` | Missing `exec` |
| 471-493 | `thrust::for_each` + `thrust::transform` | Uses `thrust::device`, Missing `exec` |
| 508-538 | `thrust::for_each` | Missing `exec` |

### Impact

- **Race conditions**: Training kernels and densification kernels run concurrently
- **Undefined behavior**: Data being read/written concurrently on different streams
- **Non-deterministic crashes**: May work sometimes, crash other times

### Recommended Fix

Replace ALL `thrust::device` with `thrust::cuda::par.on(ctx.stream)`:

```cpp
// CORRECT pattern
auto exec = thrust::cuda::par.on(ctx.stream);

thrust::transform(
  exec,  // Use exec instead of thrust::device
  m_gaussians->opacities().begin(),
  m_gaussians->opacities().end(),
  opacities.begin(),
  [] __device__ (float opacity) { return activate_opacity(opacity); }
);
```

---

## Bug 3: HIGH - Stream Mismatch in `absgs.cu:duplicate()`

**File:** `tinygs/src/strategy/absgs.cu`  
**Lines:** 99-116

### Description

The `thrust::for_each` at lines 99-116 does not use the `exec` policy with `ctx.stream`.

### Root Cause

```cpp
// Lines 99-116 - Missing exec!
thrust::for_each(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_densification_info, d_scale, d_grow_flags,
     clone_thresh, split_thresh, scale_boundary] __device__(int i) {
      // ... kernel body ...
    });
```

### Recommended Fix

```cpp
auto exec = thrust::cuda::par.on(ctx.stream);
thrust::for_each(exec,  // Add exec here
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_densification_info, d_scale, d_grow_flags,
     clone_thresh, split_thresh, scale_boundary] __device__(int i) {
      // ... kernel body ...
    });
```

---

## Bug 4: HIGH - Stream Mismatch in `improved.cu:add_noise_opacity`

**File:** `tinygs/src/strategy/improved.cu`  
**Line:** 92

### Description

The `linear_kernel` call passes `nullptr` for the stream parameter instead of `ctx.stream`.

### Root Cause

```cpp
// Line 92 - passes nullptr for stream!
linear_kernel(add_noise_opacity, 0, nullptr, N, noise_scale,
              thrust::raw_pointer_cast(m_gaussians->opacities().data()),
              noise.data());
```

### Recommended Fix

```cpp
linear_kernel(add_noise_opacity, 0, ctx.stream, N, noise_scale,
              thrust::raw_pointer_cast(m_gaussians->opacities().data()),
              noise.data());
```

---

## Bug 5: MODERATE - Stream Issues in `default.cu`

**File:** `tinygs/src/strategy/default.cu`  
**Lines:** 99-112, 151-175

### Description

Two `thrust::for_each` calls do not use the `exec` policy with `ctx.stream`.

### Root Cause

```cpp
// Lines 99-112 - Missing exec (debug block)
thrust::for_each(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_densification_info, d_gradient_values, use_absgrad = m_params.absgrad
    ] __device__(int i) { ... });

// Lines 151-175 - Missing exec
thrust::for_each(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_densification_info, d_scale, d_grow_flags, ...] __device__(int i) { ... });
```

### Recommended Fix

Add `exec` to all thrust::for_each calls:

```cpp
auto exec = thrust::cuda::par.on(ctx.stream);
thrust::for_each(exec,  // Add exec here
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [...] __device__(int i) { ... });
```

---

## Bug 6: MODERATE - Synchronous Memset Across All Strategy Files

### Description

All strategy files use synchronous `memset(0)` instead of asynchronous `memset_async(ctx.stream, 0)`. This creates unnecessary CPU-GPU synchronization points.

### Affected Locations

| File | Lines |
|------|-------|
| `default.cu` | 46, 61 |
| `fastgs.cu` | 154, 242, 267, 603 |
| `absgs.cu` | 44, 58 |
| `improved.cu` | 60, 76 |

### Root Cause

```cpp
// BAD (synchronous, blocks CPU)
ctx.densification_info->memset(0);
```

### Recommended Fix

```cpp
// GOOD (asynchronous, respects stream ordering)
ctx.densification_info->memset_async(ctx.stream, 0);
```

---

## Bug 7: LOW - Additional nullptr Stream Issues in Other Files

**Files:** 
- `tinygs/src/utils/image_format.cu` (lines 134, 138)
- `tinygs/src/loss/psnr.cu` (lines 96, 102, 107)

### Description

These files also use `nullptr` for the stream parameter in `linear_kernel` calls. While less critical than the strategy files (which run during training), these should still be fixed for consistency.

### Root Cause

```cpp
// image_format.cu:134,138
linear_kernel(half_to_float_kernel, 0, nullptr, n, src, dst);
linear_kernel(float_to_half_kernel, 0, nullptr, n, src, dst);

// psnr.cu:96,102,107
linear_kernel(psnr_squared_diff_kernel, 0, nullptr, n, ...);
```

### Recommended Fix

Pass appropriate stream parameter (may require API changes to accept stream).

---

## Use-After-Free Analysis (All Files Verified Safe)

All strategy files correctly implement the pointer capture pattern:

1. **Call `StrategyBase::on_duplicate()` FIRST** - This reallocates buffers
2. **THEN capture raw pointers** - Pointers are now valid

| File | on_duplicate() Call | Pointer Capture | Status |
|------|--------------------|-----------------|--------|
| `default.cu` | Line 224 | Lines 245-254 | ✅ SAFE |
| `fastgs.cu` | Line 385 | Lines 397-406 | ✅ SAFE |
| `absgs.cu` | Line 155 | Lines 168-177 | ✅ SAFE |
| `improved.cu` | Line 201 | Lines 213-220 | ✅ SAFE |
| `mcmc.cu` | Lines 306-310 | Lines 316-328 | ✅ SAFE |

---

## Summary Table

| # | Severity | File | Lines | Issue | Priority |
|---|----------|------|-------|-------|----------|
| 1 | **CRITICAL** | orchestrator.cu | 232-241, 1118-1140 | Stream mismatch in mean() and reorder_gaussians() | **IMMEDIATE** |
| 2 | **CRITICAL** | mcmc.cu | 221-538 | 15+ stream mismatches (thrust::device) | **IMMEDIATE** |
| 3 | **HIGH** | absgs.cu | 99-116 | Missing exec in thrust::for_each | HIGH |
| 4 | **HIGH** | improved.cu | 92 | nullptr stream in linear_kernel | HIGH |
| 5 | MODERATE | default.cu | 99-112, 151-175 | Missing exec in thrust::for_each | MEDIUM |
| 6 | MODERATE | All strategies | Various | Synchronous memset | MEDIUM |
| 7 | LOW | image_format.cu, psnr.cu | 134, 138, 96, 102, 107 | nullptr stream in linear_kernel | LOW |

---

## Recommended Action Plan

### Immediate (Critical)

1. **mcmc.cu** - Replace ALL `thrust::device` with `thrust::cuda::par.on(ctx.stream)`
2. **orchestrator.cu** - Add stream parameter to `mean()` and pass `m_major_stream` to all operations in `reorder_gaussians()`

### High Priority

3. **absgs.cu** - Add `exec` to thrust::for_each at line 99
4. **improved.cu** - Replace `nullptr` with `ctx.stream` at line 92

### Medium Priority

5. **default.cu** - Add `exec` to thrust::for_each calls
6. **All strategies** - Replace `memset(0)` with `memset_async(ctx.stream, 0)`

### Low Priority

7. **image_format.cu, psnr.cu** - Add stream parameter to linear_kernel calls

---

## Correct Pattern Reference

The correct pattern for stream usage (from `default.cu`):

```cpp
void Strategy::duplicate(const RasterizeContext& ctx) {
  // 1. Create execution policy with stream
  auto exec = thrust::cuda::par.on(ctx.stream);
  
  // 2. Use exec for ALL thrust operations
  thrust::for_each(exec, ...);
  thrust::copy(exec, ...);
  thrust::transform(exec, ...);
  
  // 3. Call on_duplicate FIRST (reallocates buffers)
  StrategyBase::on_duplicate(d_grow_indices_src, d_grow_indices_target, num_grows);
  
  // 4. THEN capture pointers (after reallocation)
  thrust::for_each(exec, 
    [means3d = thrust::raw_pointer_cast(m_gaussians->means().data()),
     scales3d = thrust::raw_pointer_cast(m_gaussians->scales().data()),
     // ... other pointers ...
    ] __device__(int i) { ... });
  
  // 5. Use async memset
  ctx.densification_info->memset_async(ctx.stream, 0);
}
```
