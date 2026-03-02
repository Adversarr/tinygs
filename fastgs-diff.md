# FastGS Implementation Differences

Comparison between:
- **Python**: `ref_impl/FastGS/utils/fast_utils.py` + `ref_impl/FastGS/scene/gaussian_model.py`
- **C++**: `tinygs/src/strategy/fastgs.cu`

---

## 1. Metric Map L1 Normalization (CRITICAL)

**Python** (`fast_utils.py:21-25, 82`):
```python
l1_loss = torch.mean(torch.abs(rendered - gt), 0)  # per-pixel mean across channels
l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))
metric_map = (l1_loss_norm > args.loss_thresh).int()
```
Min-max normalizes per-pixel L1 to [0,1] **before** thresholding.

**C++** (`fastgs.cu:75-81`):
```cpp
float l1_sum = 0.0f;
for (int c = 0; c < 3; c++) { ... l1_sum += fabsf(rendered - gt); }
metric_map[idx] = (l1_sum / 3.0f > loss_thresh) ? 1 : 0;
```
Uses raw mean L1 **without normalization**.

**Impact**: `loss_thresh` has completely different semantics. Python's threshold is relative (0-1 normalized range); C++ is absolute (raw pixel difference).

---

## 2. Photometric Loss Composition

**Python** (`fast_utils.py:27-31`):
```python
loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image, gt_image))
```
Uses weighted L1 + SSIM.

**C++** (`fastgs.cu:229-232`):
```cpp
// Reference uses (1-0.2)*L1 + 0.2*(1-SSIM); we approximate with pure L1
// since we don't have a fused_ssim reduction on GPU yet.
photometric_loss_h = thrust::reduce(...) / n_pixels;  // pure mean L1
```
Uses pure L1, no SSIM.

**Impact**: Pruning scores may differ due to missing SSIM contribution.

---

## 3. Camera Sampling Strategy

**Python** (`fast_utils.py:10-19`, `train.py:134-135`):
```python
def sampling_cameras(my_viewpoint_stack):
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))  # pop removes element
```
Samples **without replacement** from a copy of the camera list.

**C++** (`fastgs.cu:152-155`):
```cpp
for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
    const size_t idx = m_rng.next_uint(static_cast<uint32_t>(dataset_size));
    auto data = (*dataset)[idx];  // does not remove
}
```
Samples **with replacement** from the full dataset.

**Impact**: Python ensures each camera used at most once per densification; C++ may reuse cameras.

---

## 4. NaN Gradient Handling

**Python** (`gaussian_model.py:478, 482`):
```python
grad_vars = self.xyz_gradient_accum / self.denom
grad_vars[grad_vars.isnan()] = 0.0
grads_abs[grads_abs.isnan()] = 0.0
```
Explicitly handles NaN values in accumulated gradients.

**C++** (`fastgs.cu:458-460`):
```cpp
const float counter = fmaxf(d_densification_info[i].accum_counter, 1.0f);
const float grad = d_densification_info[i].accum_grad_mean2d / counter;
const float absgrad = d_densification_info[i].accum_absgrad_mean2d / counter;
```
No NaN handling.

**Impact**: Potential undefined behavior if NaN gradients occur.

---

## 5. Pruning Selection Method

**Python** (`gaussian_model.py:512-517`):
```python
padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
```
Uses **probabilistic multinomial sampling**.

**C++** (`fastgs.cu:752-770`):
```cpp
thrust::sort_by_key(exec, prune_weights.begin(), prune_weights.end(),
    sorted_indices.begin(), thrust::greater<float>());
// Take top `budget` indices
```
Uses **deterministic top-K sorting**.

**Impact**: Python's pruning is stochastic; C++ is deterministic given the same pruning scores.

---

## 6. Degenerate Rotation Check

**Python** (`gaussian_model.py:499-503`):
No degenerate rotation check in pruning criteria.

**C++** (`fastgs.cu:702`):
```cpp
bool not_degenerate = sum(abs(rotation[i])) > FLT_EPSILON;
```
Additional check to prune Gaussians with degenerate (zero) rotations.

**Impact**: C++ removes more Gaussians with invalid rotations.

---

## 7. Importance Threshold Configurability

**Python** (`gaussian_model.py:494`):
```python
metric_mask = importance_score > 5  # hard-coded
```

**C++** (`fastgs.hpp:70`):
```cpp
float m_importance_threshold = 5.0f;  // configurable via JSON
```

**Impact**: Same default, but C++ allows runtime configuration.

---

## Summary Table

| Aspect | Python | C++ | Severity |
|--------|--------|-----|----------|
| L1 normalization before threshold | Yes (min-max) | No | **Critical** |
| SSIM in photometric loss | Yes | No | Medium |
| Camera sampling | Without replacement | With replacement | Low |
| NaN gradient handling | Yes | No | Medium |
| Pruning selection | Multinomial (stochastic) | Top-K (deterministic) | Low |
| Degenerate rotation prune | No | Yes | Low |
| Importance threshold | Hard-coded (5) | Configurable (default 5) | None |

---

## Recommendations

1. **High Priority**: Implement L1 normalization in C++ to match Python's semantics, OR document that `loss_thresh` values are not portable between implementations.

2. **Medium Priority**: Add SSIM to photometric loss in C++, or document the approximation.

3. **Low Priority**: Consider matching camera sampling (without replacement) and/or NaN handling for exact behavioral parity.
