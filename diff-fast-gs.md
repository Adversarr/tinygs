# Comprehensive Comparison: tinygs vs Reference Implementations

This document provides a detailed comparison between **tinygs** (our C++/CUDA implementation) and three reference implementations:
- **gaussian-splatting** (Original 3DGS from INRIA)
- **FastGS** (100-second training framework)
- **AbsGS** (Absolute gradient-based densification)

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Language & Infrastructure](#2-language--infrastructure)
3. [Rasterizer Implementation](#3-rasterizer-implementation)
4. [Densification Strategies](#4-densification-strategies)
5. [Loss Functions](#5-loss-functions)
6. [Optimizer Support](#6-optimizer-support)
7. [Memory Management](#7-memory-management)
8. [Training Pipeline](#8-training-pipeline)
9. [Feature Matrix](#9-feature-matrix)
10. [Implementation Details](#10-implementation-details)
11. [Deep Implementation Comparison](#11-deep-implementation-comparison)
12. [Hyperparameter Comparison Table](#12-hyperparameter-comparison-table)
13. [Key Thresholds and Their Impact](#13-key-thresholds-and-their-impact)
14. [Sparse Adam Implementation](#14-sparse-adam-implementation)
15. [Memory Layout Comparison](#15-memory-layout-comparison)
16. [Split Algorithm Details](#16-split-algorithm-details)
17. [Training Loop Timing](#17-training-loop-timing)
18. [API Differences](#18-api-differences)
19. [Code Reference Locations](#19-code-reference-locations)
20. [Summary of Differences](#20-summary-of-differences)
21. [Additional FastGS Parameters](#21-additional-fastgs-parameters)
22. [Additional AbsGS Parameters](#22-additional-absgs-parameters)
23. [Rasterizer Implementation Details](#23-rasterizer-implementation-details)
24. [Backward Pass Implementation](#24-backward-pass-implementation)
25. [SSIM Implementation Details](#25-ssim-implementation-details)
26. [Performance Considerations](#26-performance-considerations)
27. [Debugging and Profiling](#27-debugging-and-profiling)
28. [Precise Code Reference Locations](#28-precise-code-reference-locations)
29. [AbsGS Unique Features Not in Other Implementations](#29-absgs-unique-features-not-in-other-implementations)
30. [FastGS Optimizer Scheduling Details](#30-fastgs-optimizer-scheduling-details)
31. [tinygs Implementation Specifics](#31-tinygs-implementation-specifics)
32. [Training Loop Timeline Comparison](#32-training-loop-timeline-comparison)
33. [File Reference Summary](#33-file-reference-summary)
34. [Fused SSIM Implementation Comparison](#34-fused-ssim-implementation-comparison)
35. [Sparse Adam CUDA Implementation](#35-sparse-adam-cuda-implementation)
36. [Detailed Backward Pass Gradient Flow](#36-detailed-backward-pass-gradient-flow)
37. [Metric Mode Implementation Details](#37-metric-mode-implementation-details)
38. [Gaussian Weight Tracking (AbsGS Unique Feature)](#38-gaussian-weight-tracking-absgs-unique-feature)
39. [Depth Regularization (Original 3DGS Only)](#39-depth-regularization-original-3dgs-only)
40. [Exposure Compensation (Original 3DGS Only)](#40-exposure-compensation-original-3dgs-only)
41. [Training Differences Summary Table](#41-training-differences-summary-table)
42. [Code Architecture Comparison](#42-code-architecture-comparison)
43. [Performance Optimization Techniques](#43-performance-optimization-techniques)
44. [Hyperparameter Defaults Comparison](#44-hyperparameter-defaults-comparison)
45. [Key Implementation Files Summary](#45-key-implementation-files-summary)
46. [Known Limitations and Missing Features](#46-known-limitations-and-missing-features)
47. [Recommended Configuration by Use Case](#47-recommended-configuration-by-use-case)
48. [References and Sources](#48-references-and-sources)
49. [Detailed Forward Pass Blending Algorithm](#49-detailed-forward-pass-blending-algorithm)
50. [Exact SSIM Gaussian Kernel Weights](#50-exact-ssim-gaussian-kernel-weights)
51. [Camera Model Implementation Details](#51-camera-model-implementation-details)
52. [Detailed Backward Pass Gradient Equations](#52-detailed-backward-pass-gradient-equations)
53. [Spherical Harmonics Evaluation Details](#53-spherical-harmonics-evaluation-details)
54. [Tile-Based Rendering Pipeline Details](#54-tile-based-rendering-pipeline-details)
55. [Detailed Densification Criteria](#55-detailed-densification-criteria)
56. [Additional Algorithm Constants](#56-additional-algorithm-constants)
57. [Numerical Stability Considerations](#57-numerical-stability-considerations)
58. [Implementation-Specific Optimizations](#58-implementation-specific-optimizations)
59. [Complete Hyperparameter Reference](#59-complete-hyperparameter-reference)
60. [Code Size and Complexity Metrics](#60-code-size-and-complexity-metrics)
61. [Edge Cases and Special Handling](#61-edge-cases-and-special-handling)
62. [Debugging and Validation](#62-debugging-and-validation)
63. [Future Extension Points](#63-future-extension-points)
64. [Gaussian Initialization Methods](#64-gaussian-initialization-methods)
65. [Python Binding Implementation Details](#65-python-binding-implementation-details)
66. [Viewspace Point Tensor Gradient Channels](#66-viewspace-point-tensor-gradient-channels)
67. [Memory Buffer Management](#67-memory-buffer-management)
68. [Additional FastGS Implementation Details](#68-additional-fastgs-implementation-details)
69. [Additional AbsGS Implementation Details](#69-additional-absgs-implementation-details)
70. [Detailed Comparison of Gradient Accumulation](#70-detailed-comparison-of-gradient-accumulation)
71. [Summary of All Key Differences](#71-summary-of-all-key-differences)
72. [MCMC Strategy Implementation (tinygs Exclusive)](#72-mcmc-strategy-implementation-tinygs-exclusive)
73. [Improved Strategy Implementation (tinygs Exclusive)](#73-improved-strategy-implementation-tinygs-exclusive)
74. [Additional Loss Functions (tinygs Exclusive)](#74-additional-loss-functions-tinygs-exclusive)
75. [tinygs Optimizer Architecture Details](#75-tinygs-optimizer-architecture-details)
76. [tinygs Rasterizer Architecture Details](#76-tinygs-rasterizer-architecture-details)
77. [DensificationInfo Structure](#77-densificationinfo-structure)
78. [Strategy Parameters Summary](#78-strategy-parameters-summary)
79. [Python vs tinygs: Detailed API Comparison](#79-python-vs-tinygs-detailed-api-comparison)
80. [Code Size Comparison](#80-code-size-comparison)
81. [Configuration Files Comparison](#81-configuration-files-comparison)
82. [Scene-Specific Hyperparameter Tuning (FastGS Exclusive)](#82-scene-specific-hyperparameter-tuning-fastgs-exclusive)
83. [Undocumented Features and Minor Changes (AbsGS)](#83-undocumented-features-and-minor-changes-absgs)
84. [Additional Implementation Details](#84-additional-implementation-details)
85. [Performance Characteristics](#85-performance-characteristics)
86. [Dataset-Specific Behaviors](#86-dataset-specific-behaviors)
87. [Implementation-Specific Constants](#87-implementation-specific-constants)
88. [Code Quality and Maintenance](#88-code-quality-and-maintenance)
89. [Testing and Validation](#89-testing-and-validation)
90. [Differences in Training Scripts](#90-differences-in-training-scripts)
91. [File Organization Differences](#91-file-organization-differences)
92. [Build and Dependency Management](#92-build-and-dependency-management)
93. [Known Limitations by Implementation](#93-known-limitations-by-implementation)
94. [Summary of All Discovered Details](#94-summary-of-all-discovered-details)
95. [Recommendations for tinygs Users](#95-recommendations-for-tinygs-users)
96. [tinygs Exclusive Features (Not in Python References)](#96-tinygs-exclusive-features-not-in-python-references)
97. [Configuration File Format Comparison](#97-configuration-file-format-comparison)
98. [Implementation Differences Summary](#98-implementation-differences-summary)
99. [Practical Training Recommendations](#99-practical-training-recommendations)
100. [Final Summary](#100-final-summary)

---

## 1. Architecture Overview

### gaussian-splatting (Original 3DGS)
- **Language**: Python with CUDA extensions
- **Architecture**: PyTorch-based optimizer with custom CUDA rasterizer
- **Structure**:
  ```
  gaussian-splatting/
  ├── train.py              # Main training loop
  ├── scene/
  │   ├── gaussian_model.py # Gaussian parameters & densification
  │   ├── cameras.py        # Camera model
  │   └── dataset_readers.py
  ├── gaussian_renderer/
  │   └── __init__.py       # Rendering interface
  └── submodules/
      ├── diff-gaussian-rasterization/  # CUDA rasterizer
      └── simple-knn/                   # KNN for initialization
  ```

### FastGS
- **Language**: Python with CUDA extensions
- **Architecture**: Extends 3DGS with multi-view metric scoring
- **Key Innovation**: 100-second training via:
  - Multi-view consistency metrics for densification control
  - Budget-based pruning
  - Optimized learning rate schedules
- **Structure**:
  ```
  FastGS/
  ├── train.py              # Modified training with metric scoring
  ├── scene/
  │   └── gaussian_model.py # densify_and_prune_fastgs(), final_prune_fastgs()
  ├── utils/
  │   └── fast_utils.py     # compute_gaussian_score_fastgs()
  └── submodules/
      └── diff-gaussian-rasterization-fastgs/  # Metric mode support
  ```

### AbsGS
- **Language**: Python with CUDA extensions
- **Architecture**: Introduces absolute gradient for split decisions
- **Key Innovation**: Separates gradient computation:
  - Standard gradient (norm of 2D position gradient) → for cloning
  - Absolute gradient (separate tracking) → for splitting
- **Structure**:
  ```
  AbsGS/
  ├── train.py              # Training with abs gradient tracking
  ├── scene/
  │   └── gaussian_model.py # xyz_gradient_accum_abs for split
  └── submodules/
      └── diff-gaussian-rasterization-abs/  # Abs gradient rasterizer
  ```

### tinygs
- **Language**: C++20 with CUDA 12.4+
- **Architecture**: Pure native implementation with modular design
- **Key Features**:
  - Multiple rasterizer backends (default, fastgs_ours, 3dgs_accel, fastgs_ours_fp16)
  - Multiple densification strategies (default, fastgs, absgs, mcmc, improved)
  - Native fused SSIM loss in CUDA
  - FP16 support
- **Structure**:
  ```
  tinygs/
  ├── tinygs/src/
  │   ├── core/           # Gaussian primitives, camera models
  │   ├── cuda/           # CUDA utilities, memory management
  │   ├── rasterizer/     # Multiple rasterizer implementations
  │   │   ├── 3dgs_accel/    # Original 3DGS algorithm
  │   │   ├── fastgs_ours/   # Custom FastGS implementation (FP32)
  │   │   └── fastgs_ours_fp16/  # FP16 variant
  │   ├── optim/          # Optimizers (adam, adamw, sgd)
  │   ├── loss/           # Loss functions (l1, fused_ssim)
  │   ├── strategy/       # Densification strategies
  │   └── dataloader/     # Data loading utilities
  └── apps/               # Executables (config_train, export_default)
  ```

---

## 2. Language & Infrastructure

| Aspect | gaussian-splatting | FastGS | AbsGS | tinygs |
|--------|-------------------|--------|-------|--------|
| **Primary Language** | Python | Python | Python | C++20 |
| **CUDA Integration** | PyBind11 | PyBind11 | PyBind11 | Native CUDA |
| **Build System** | pip/setuptools | pip/setuptools | pip/setuptools | CMake 3.28+ |
| **Minimum CUDA** | 11.x | 11.x | 11.x | 12.4+ |
| **GPU Compute** | 7.0+ | 7.0+ | 7.0+ | 7.0+ |
| **Dependencies** | PyTorch, plyfile, tqdm | PyTorch, fused_ssim | PyTorch | OpenCV, spdlog, nlohmann_json, glm, NVTX3 |

### tinygs Advantages
- **No Python Runtime**: Pure C++/CUDA execution eliminates Python overhead
- **Compile-time Optimization**: Template metaprogramming for kernel specialization
- **Memory Efficiency**: Direct GPU memory management without Python allocations
- **NVTX Profiling**: Built-in profiling ranges for all kernels

---

## 3. Rasterizer Implementation

### 3.1 Algorithm Comparison

#### gaussian-splatting Rasterizer
```python
# Key pipeline steps:
1. Preprocess: Project 3D Gaussians → 2D screen space
2. Duplicate with keys: Create (tile_id, depth) pairs for each tile overlap
3. Sort: Radix sort by tile then depth
4. Identify ranges: Find start/end indices per tile
5. Render: Alpha blending per tile in parallel
```

#### FastGS Rasterizer Extension
```python
# Additional features:
- metric_map parameter: Binary mask of high-loss pixels
- accum_metric_counts output: Per-Gaussian count of high-loss overlaps
- mult parameter: Tile bounding box multiplier for compact culling
```
Key code from `gaussian_renderer/__init__.py`:
```python
render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, 
                           get_flag = get_flag, metric_map = metric_map)
accum_loss_counts = render_pkg["accum_metric_counts"]
```

#### tinygs Rasterizer (fastgs_ours)
```cpp
// Forward pass stages (from forward.cu):
1. preprocess_cu: Project Gaussians, compute conic, colors, bounds
2. Sort depth keys: CUB radix sort by depth
3. apply_depth_ordering: Compute tile touch counts per Gaussian
4. Exclusive sum scan: Compute instance offsets
5. create_instances: Emit (tile_id, depth) key-value pairs
6. Sort tiles: CUB radix sort by tile ID
7. extract_instance_ranges: Find start/end per tile
8. extract_bucket_counts: Count buckets per tile
9. Inclusive sum: Bucket offsets
10. blend_cu: Tile-based alpha blending with optional metric mode
```

### 3.2 Key Implementation Differences

| Feature | gaussian-splatting | FastGS | AbsGS | tinygs |
|---------|-------------------|--------|-------|--------|
| **Tile Size** | 16×16 | 16×16 | 16×16 | 16×16 |
| **Sort Backend** | CUB | CUB | CUB | CUB |
| **Metric Mode** | ❌ | ✅ | ❌ | ✅ |
| **Bucket-based Blending** | ❌ | ❌ | ❌ | ✅ |
| **FP16 Support** | ❌ | ❌ | ❌ | ✅ |
| **Tiled Image Layout** | ❌ | ❌ | ❌ | ✅ |
| **Dual Stream Execution** | ❌ | ❌ | ❌ | ✅ |

### 3.3 tinygs Tiled Image Layout
tinygs uses a tiled memory layout for images (CHW format with 8×8 tiles):
```cpp
// From helper_math.h / common_device.cuh
constexpr int kImageTile = 8;
constexpr int kImageTileLog2 = 3;
constexpr int kImageTileMask = 7;

inline uint get_linear_index_tiled(uint y, uint x, uint width_in_tile) {
    uint tile_y = y >> kImageTileLog2;
    uint tile_x = x >> kImageTileLog2;
    uint in_tile_y = y & kImageTileMask;
    uint in_tile_x = x & kImageTileMask;
    return ((tile_y * width_in_tile + tile_x) << (2 * kImageTileLog2))
         | (in_tile_y << kImageTileLog2) | in_tile_x;
}
```
This improves memory coalescing for texture access patterns.

### 3.4 Metric Mode Implementation

**FastGS (Python)**:
```python
# From utils/fast_utils.py
metric_map = (l1_loss_norm > args.loss_thresh).int()
render_pkg = render_fastgs(cam, gaussians, pipe, bg, mult, 
                           get_flag=True, metric_map=metric_map)
importance_score = render_pkg["accum_metric_counts"] / len(camlist)
```

**tinygs (CUDA)**:
```cpp
// From kernels_forward.cuh (blend_cu with metric_mode=true)
if (metric_mode) {
    // Per-pixel metric accumulation
    int metric_val = metric_map[pixel_idx];
    atomicAdd(&metric_counts[primitive_idx], metric_val);
}
```

---

## 4. Densification Strategies

### 4.1 Original 3DGS Strategy (default)

**Algorithm**:
```
For each densification interval (every 100 iterations, 500-15000):
  1. Compute average gradient: grads = xyz_gradient_accum / denom
  2. Clone: grads >= threshold AND scale <= percent_dense * extent
  3. Split: grads >= threshold AND scale > percent_dense * extent
  4. Prune: opacity < 0.005 OR max_radii2D > threshold
```

**Code (gaussian-splatting/scene/gaussian_model.py)**:
```python
def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
    grads = self.xyz_gradient_accum / self.denom
    grads[grads.isnan()] = 0.0
    self.densify_and_clone(grads, max_grad, extent)
    self.densify_and_split(grads, max_grad, extent)
    prune_mask = (self.get_opacity < min_opacity).squeeze()
    # ... additional pruning criteria
    self.prune_points(prune_mask)
```

### 4.2 AbsGS Strategy (absgs)

**Key Innovation**: Separate gradient tracking for clone vs split decisions.

**Algorithm**:
```
Track two gradient accumulators:
  - xyz_gradient_accum: Norm of 2D position gradient (for cloning)
  - xyz_gradient_accum_abs: Separate tracking (for splitting)

Clone: norm(xyz_gradient_accum/denom) >= grad_thresh AND small scale
Split: norm(xyz_gradient_accum_abs/denom) >= grad_abs_thresh AND large scale
```

**Code (AbsGS/scene/gaussian_model.py)**:
```python
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    # Standard gradient (first 2 components)
    self.xyz_gradient_accum[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    # Absolute gradient (last 2 components)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
    self.denom[update_filter] += 1
```

**tinygs Implementation** (from `strategy/absgs.hpp`):
```cpp
// Uses separate gradient channels in DensificationInfo
// Forward kernel computes both:
// - grad_mean2d: standard gradient (for clone)
// - absgrad_mean2d: absolute gradient (for split)
```

### 4.3 FastGS Strategy (fastgs)

**Key Innovations**:
1. Multi-view metric scoring
2. Budget-based pruning
3. Final aggressive pruning phase

**Algorithm**:
```
During densification (500-15000, every 100 iterations):
  1. Sample 10 random cameras (without replacement)
  2. For each camera:
     a. Render image
     b. Compute normalized L1 loss per pixel
     c. Threshold to create metric_map (loss > loss_thresh)
     d. Accumulate per-Gaussian metric counts and photometric scores
  3. Compute importance_score = metric_counts / num_cameras
  4. Compute pruning_score = normalized photometric_score
  5. Clone: grad >= thresh AND importance > 5 AND small scale
  6. Split: abs_grad >= thresh AND importance > 5 AND large scale
  7. Prune: low opacity OR large size, with budget sampling

Final pruning (every 3000, 15000-30000):
  1. Re-compute pruning scores
  2. Prune: opacity < 0.1 OR pruning_score > 0.9
```

**Code (FastGS/scene/gaussian_model.py)**:
```python
def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, 
                             args, importance_score=None, pruning_score=None):
    grad_vars = self.xyz_gradient_accum / self.denom
    grads_abs = self.xyz_gradient_accum_abs / self.denom
    
    # Gradient qualifiers
    grad_qualifiers = torch.norm(grad_vars, dim=-1) >= args.grad_thresh
    grad_qualifiers_abs = torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh
    
    # Scale qualifiers
    clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense * extent
    split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense * extent
    
    # Multi-view importance filter
    metric_mask = importance_score > 5
    
    self.densify_and_clone_fastgs(metric_mask, clone_qualifiers & grad_qualifiers)
    self.densify_and_split_fastgs(metric_mask, split_qualifiers & grad_qualifiers_abs)
    
    # Budget-based pruning
    scores = 1 - pruning_score
    to_remove = torch.sum(prune_mask)
    remove_budget = int(0.5 * to_remove)
    # Multinomial sampling for removal...
```

**tinygs Implementation** (from `strategy/fastgs.hpp`):
```cpp
class FastGSStrategy : public StrategyBase {
  // Hyper-parameters
  float m_absgrad_threshold = 0.0012f;
  float m_loss_thresh = 0.1f;
  int m_metric_num_cameras = 10;
  float m_importance_threshold = 5.0f;
  float m_prune_budget_ratio = 0.5f;
  float m_final_prune_score_threshold = 0.9f;
  float m_final_prune_opacity_threshold = 0.1f;
  int m_final_prune_start = 18000;
  int m_final_prune_every = 3000;
  
  // Accumulated scores
  thrust::device_vector<float> m_importance_score;
  thrust::device_vector<float> m_pruning_score;
  
  void compute_gaussian_score(const RasterizeContext& ctx, bool densify);
  void final_prune(const RasterizeContext& ctx);
};
```

### 4.4 Strategy Comparison Matrix

| Feature | default | absgs | fastgs | mcmc | improved |
|---------|---------|-------|--------|------|----------|
| **Separate Clone/Split Gradients** | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Multi-view Metric Scoring** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Budget-based Pruning** | ❌ | ❌ | ✅ | ❌ | ✅ |
| **Final Aggressive Pruning** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Opacity Reset** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Opacity Reduce** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Weight-based Pruning** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Initial Prune** | ❌ | ✅ | ❌ | ❌ | ❌ |

---

## 5. Loss Functions

### 5.1 SSIM Implementation

#### Python References (PyTorch-based)
All three Python implementations use PyTorch convolution for SSIM:
```python
# From loss_utils.py
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    sigma1_sq = F.conv2d(img1 * img1, window, ...) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, ...) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, ...) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / 
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
```

#### FastGS Fused SSIM
FastGS uses `fused_ssim` package (separate CUDA extension):
```python
from fused_ssim import fused_ssim as fast_ssim
ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
```

#### tinygs Native Fused SSIM
Implemented from scratch in CUDA with FP32 and FP16 variants:
```cpp
// From loss/fused_ssim.cu

// FP32 forward kernel
__global__ void fused_ssim_cuda_fp32(
    int H, int W, float C1, float C2, float scale,
    const float* img1, const float* img2,
    float* ssim_map,
    float* dm_dmu1,      // Partial derivatives for backward
    float* dm_dsigma1_sq,
    float* dm_dsigma12
);

// Two-pass separable convolution:
// 1. Horizontal: 11×1 Gaussian kernel
// 2. Vertical: 1×11 Gaussian kernel
// 3. Compute SSIM and store partial derivatives
```

Key optimizations:
- **Shared memory tiling**: 16×16 blocks with 5-pixel halo
- **Half2 vectorization**: Load two pixels at once
- **Warp-level reduction**: Efficient partial sum reduction
- **Tiled image layout**: Memory coalescing optimization

### 5.2 Combined Loss

**Standard formulation** (all implementations):
```python
loss = (1 - lambda_dssim) * L1_loss + lambda_dssim * (1 - SSIM)
# Default: lambda_dssim = 0.2
```

**tinygs Loss Context**:
```cpp
struct LossContext {
  Buffer pred;      // Rendered image
  Buffer target;    // Ground truth image
  Buffer loss;      // Output loss map
  Buffer grad;      // Output gradient
  cudaStream_t stream;
  DataType data_type;
};
```

---

## 6. Optimizer Support

### 6.1 Reference Implementations

| Implementation | Adam | Sparse Adam | AdamW | SGD |
|---------------|------|-------------|-------|-----|
| gaussian-splatting | ✅ | ✅ (optional) | ❌ | ❌ |
| FastGS | ✅ | ✅ (optional) | ❌ | ❌ |
| AbsGS | ✅ | ❌ | ❌ | ❌ |

**Sparse Adam** (gaussian-splatting, FastGS):
```python
# Only updates Gaussians visible in current view
if opt.optimizer_type == "sparse_adam":
    visible = radii > 0
    gaussians.optimizer.step(visible, radii.shape[0])
```

### 6.2 tinygs Optimizers

```cpp
// From optim/ directory
class OptimizerBase {
  virtual void step(cudaStream_t stream) = 0;
  virtual void step_sparse(const bool* visible, int n, cudaStream_t stream) = 0;
  virtual void reset() = 0;
  virtual void reset_opacity() = 0;
  virtual void remove(const char* kept_flag, int num_kept) = 0;
  virtual void duplicate(const int* indices, int* new_indices, int num) = 0;
};

// Concrete implementations:
class AdamOptimizer : public OptimizerBase;
class AdamWOptimizer : public OptimizerBase;
class SGDOptimizer : public OptimizerBase;
```

### 6.3 Optimizer Step Scheduling

**FastGS Custom Schedule**:
```python
def optimizer_step(self, iteration):
    if iteration <= 15000:
        self.optimizer.step()
        if iteration % 16 == 0:
            self.shoptimizer.step()  # Separate SH optimizer
    elif iteration <= 20000:
        if iteration % 32 == 0:
            self.optimizer.step()
            self.shoptimizer.step()
    else:
        if iteration % 64 == 0:
            self.optimizer.step()
            self.shoptimizer.step()
```

This reduces optimization frequency in later stages, trading quality for speed.

---

## 7. Memory Management

### 7.1 Reference Implementations (Python)

```python
# PyTorch manages GPU memory automatically
# Manual intervention for large datasets:
if args.data_device == "cpu":
    # Store images on CPU, transfer to GPU as needed
    
# Explicit cache clearing:
torch.cuda.empty_cache()
```

### 7.2 tinygs Memory Architecture

```cpp
// GPUBuffer: RAII wrapper for CUDA memory
template <typename T>
class GPUBuffer {
  T* m_data = nullptr;
  size_t m_size = 0;
  cudaStream_t m_stream;
public:
  void ensure(size_t size);
  void resize(size_t size);
  void free();
};

// Buffer pools for rasterizer (from fastgs_ours/forward.cu)
struct PerPrimitiveBuffers {
  DoubleBuffer<uint> primitive_indices;
  DoubleBuffer<int> depth_keys;
  uint* n_touched_tiles;
  uint2* screen_bounds;
  float2* mean2d;
  float4* conic_opacity;
  float3* color;
  uint* offset;
  // CUB workspace...
};

struct PerTileBuffers {
  uint2* instance_ranges;
  uint* n_buckets;
  uint* bucket_offsets;
  // ...
};

struct PerBucketBuffers {
  uint* tile_index;
  float4* color_transmittance;  // For backward pass
};
```

### 7.3 Memory Layout Differences

| Aspect | Python References | tinygs |
|--------|------------------|--------|
| **Allocation Strategy** | PyTorch allocator | Custom GPUBuffer pools |
| **Double Buffering** | ❌ | ✅ (for ping-pong sorting) |
| **Zero-copy Host Access** | ❌ | ✅ (for visibility counts) |
| **Tiled Image Storage** | ❌ (CHW linear) | ✅ (tiled CHW) |
| **FP16 Support** | Partial | Full (fastgs_ours_fp16) |

---

## 8. Training Pipeline

### 8.1 Training Loop Comparison

#### Gaussian-splatting Training Loop
```python
for iteration in range(first_iter, opt.iterations + 1):
    # 1. Update learning rate
    gaussians.update_learning_rate(iteration)
    
    # 2. Increase SH degree every 1000 iterations
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()
    
    # 3. Pick random camera
    viewpoint_cam = viewpoint_stack.pop(rand_idx)
    
    # 4. Render
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    
    # 5. Compute loss
    Ll1 = l1_loss(image, gt_image)
    loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - ssim)
    
    # 6. Backward
    loss.backward()
    
    # 7. Densification (if in range)
    if iteration < opt.densify_until_iter:
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        if iteration % opt.densification_interval == 0:
            gaussians.densify_and_prune(...)
        if iteration % opt.opacity_reset_interval == 0:
            gaussians.reset_opacity()
    
    # 8. Optimizer step
    gaussians.optimizer.step()
```

#### FastGS Training Loop
```python
for iteration in range(first_iter, opt.iterations + 1):
    # ... same rendering and loss ...
    
    # Densification with multi-view metrics
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        camlist = sampling_cameras(viewpoint_stack)  # 10 random cameras
        importance_score, pruning_score = compute_gaussian_score_fastgs(
            camlist, gaussians, pipe, bg, opt, DENSIFY=True)
        gaussians.densify_and_prune_fastgs(...)
    
    # Final pruning phase (15k-30k, every 3k)
    if iteration % 3000 == 0 and iteration > 15000 and iteration < 30000:
        _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, ...)
        gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)
    
    # Optimizer step with custom schedule
    gaussians.optimizer_step(iteration)
```

### 8.2 Default Hyperparameters

| Parameter | gaussian-splatting | FastGS | AbsGS |
|-----------|-------------------|--------|-------|
| `iterations` | 30,000 | 30,000 | 30,000 |
| `position_lr_init` | 0.00016 | 0.00016 | 0.00016 |
| `position_lr_final` | 0.0000016 | 0.0000016 | 0.0000016 |
| `feature_lr` | 0.0025 | 0.0025 (low) | 0.0025 |
| `highfeature_lr` | - | 0.005 | - |
| `opacity_lr` | 0.05 | 0.025 | 0.05 |
| `scaling_lr` | 0.005 | 0.005 | 0.005 |
| `rotation_lr` | 0.001 | 0.001 | 0.001 |
| `densify_from_iter` | 500 | 500 | 500 |
| `densify_until_iter` | 15,000 | 15,000 | 15,000 |
| `densification_interval` | 100 | 100 | 100 |
| `densify_grad_threshold` | 0.0002 | 0.0002 | 0.0002 |
| `densify_grad_abs_threshold` | - | 0.0012 | 0.0004 |
| `opacity_reset_interval` | 3,000 | 3,000 | 3,000 |
| `percent_dense` | 0.01 | 0.001 | 0.001 |
| `lambda_dssim` | 0.2 | 0.2 | 0.2 |

### 8.3 FastGS-Specific Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `loss_thresh` | 0.1 | L1 threshold for metric map |
| `grad_abs_thresh` | 0.0012 | Split gradient threshold |
| `dense` | 0.001 | Scale boundary for clone/split |
| `mult` | 0.5 | Tile bounding box multiplier |

---

## 9. Feature Matrix

### 9.1 Core Features

| Feature | gaussian-splatting | FastGS | AbsGS | tinygs |
|---------|-------------------|--------|-------|--------|
| **Training** | ✅ | ✅ | ✅ | ✅ |
| **Rendering** | ✅ | ✅ | ✅ | ✅ |
| **PLY Export** | ✅ | ✅ | ✅ | ✅ |
| **PLY Import** | ✅ | ✅ | ✅ | ✅ |
| **COLMAP Dataset** | ✅ | ✅ | ✅ | ✅ |
| **Evaluation Split** | ✅ | ✅ | ✅ | ✅ |

### 9.2 Advanced Features

| Feature | gaussian-splatting | FastGS | AbsGS | tinygs |
|---------|-------------------|--------|-------|--------|
| **Sparse Adam** | ✅ | ✅ | ❌ | ✅ |
| **Depth Regularization** | ✅ | ❌ | ❌ | ❌ |
| **Exposure Compensation** | ✅ | ❌ | ❌ | ❌ |
| **Anti-aliasing** | ✅ | ❌ | ❌ | ❌ |
| **FP16 Training** | ❌ | ❌ | ❌ | ✅ |
| **Multiple Rasterizers** | ❌ | ❌ | ❌ | ✅ |
| **Multiple Strategies** | ❌ | ❌ | ❌ | ✅ |
| **NVTX Profiling** | ❌ | ❌ | ❌ | ✅ |
| **Config-driven Training** | ❌ | ❌ | ❌ | ✅ |

### 9.3 Densification Strategies

| Strategy | gaussian-splatting | FastGS | AbsGS | tinygs |
|----------|-------------------|--------|-------|--------|
| **Default (3DGS)** | ✅ | - | - | ✅ |
| **AbsGS** | - | - | ✅ | ✅ |
| **FastGS** | - | ✅ | - | ✅ |
| **MCMC** | - | - | - | ✅ |
| **Improved** | - | - | - | ✅ |

---

## 10. Implementation Details

### 10.1 Spherical Harmonics

All implementations use the same SH representation:
- DC coefficient (degree 0): 3 values
- Rest coefficients (degrees 1-3): 45 values (3 × 15)
- Total: 48 values per Gaussian (3 channels × 16 coefficients)

**tinygs SH Storage**:
```cpp
// Separate buffers for DC and rest
float* sh0;  // DC (N × 1 × 3)
float* sh1;  // Degree 1 (N × 3 × 3)
float* sh2;  // Degree 2 (N × 5 × 3)
float* sh3;  // Degree 3 (N × 7 × 3)
```

### 10.2 Gaussian Parameterization

| Parameter | Storage | Activation |
|-----------|---------|------------|
| Position | float3 | Raw |
| Scale | float3 | exp() |
| Rotation | float4 (quaternion) | normalize() |
| Opacity | float | sigmoid() |
| SH features | float | Raw |

### 10.3 Camera Model

All implementations support:
- Pinhole camera model
- Radial distortion (handled in preprocessing)
- World-to-camera transformation matrix
- Camera center (for view-dependent shading)

### 10.4 Rendering Equation

```cpp
// Alpha blending (all implementations)
for each Gaussian g in sorted order (back to front):
    alpha = opacity * exp(-0.5 * (dx² * conic.x + 2*dx*dy*conic.y + dy² * conic.z))
    T = T * (1 - alpha)  // Transmittance
    color += T * alpha * g.color
    
// Where conic is the inverse 2D covariance matrix
```

### 10.5 Backward Pass Gradients

The backward pass computes gradients for:
- `dL/dmeans3D`: 3D position gradients
- `dL/dscales`: Scale gradients  
- `dL/drotations`: Rotation quaternion gradients
- `dL/dopacity`: Opacity gradients
- `dL/dsh`: Spherical harmonics gradients

**tinygs additionally supports**:
- `absgrad_mean2d`: Absolute gradient for AbsGS/FastGS strategies
- `dL/dw2c`: Camera pose gradients (optional)

---

## Summary

### Key Advantages of tinygs

1. **Performance**: Native C++/CUDA eliminates Python interpreter overhead
2. **Modularity**: Pluggable rasterizers and strategies
3. **Memory Efficiency**: Custom memory management, FP16 support
4. **Profiling**: Built-in NVTX ranges for all operations
5. **Extensibility**: Clean interfaces for adding new strategies/rasterizers

### Key Innovations from References

1. **From FastGS**: Multi-view metric scoring for controlled densification
2. **From AbsGS**: Absolute gradient for improved split decisions
3. **From gaussian-splatting**: Sparse Adam optimizer, depth regularization

### Missing Features in tinygs (vs references)

1. **Depth Regularization**: No depth map prior support
2. **Exposure Compensation**: No per-image exposure optimization
3. **Anti-aliasing**: No EWA filter integration
4. **GUI Viewer**: No network/real-time viewer

### Recommended Strategy

For fastest training with comparable quality:
- Use **FastGS strategy** with `fastgs_ours_fp16` rasterizer
- Enable FP16 training for memory efficiency
- Adjust `loss_thresh` and `importance_threshold` for scene complexity

---

## 11. Deep Implementation Comparison

### 11.1 AbsGS Homodirectional Gradient (Critical Innovation)

The **key difference** in AbsGS is the homodirectional gradient computation in the backward pass.

**Original 3DGS Backward** (ref_impl/gaussian-splatting):
```cpp
// Standard signed gradient only
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
// dL_dmean2D is float2
```

**AbsGS Backward** (ref_impl/AbsGS/submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/backward.cu:545-550):
```cpp
// Standard gradient (signed)
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

// NEW: Homodirectional Gradient - absolute value of gradient
atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * ddelx_dx));
atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * ddely_dy));
// dL_dmean2D is float4 (not float2!)
```

**Python Binding Change** (ref_impl/AbsGS/submodules/diff-gaussian-rasterization-abs/rasterize_points.cu:154):
```cpp
torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());  // 4 components, not 2!
```

**Gradient Accumulation** (ref_impl/AbsGS/scene/gaussian_model.py:479-482):
```python
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    # Standard gradient: L2 norm of first 2 components (indices 0:2)
    self.xyz_gradient_accum[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    
    # Absolute gradient: L2 norm of LAST 2 components (indices 2:4)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
```

**tinygs Implementation** (tinygs/src/rasterizer/fastgs_ours/kernels_backward.cuh:677, 691-692):
```cpp
// Accumulate absolute gradient during blend backward
absdL_dmean2d_accum += make_float2(fabsf(dL_dmean2d.x), fabsf(dL_dmean2d.y));
// ...
if (absgrad_mean2d != nullptr) {
    atomicAdd(&absgrad_mean2d[primitive_idx].x, absdL_dmean2d_accum.x);
    atomicAdd(&absgrad_mean2d[primitive_idx].y, absdL_dmean2d_accum.y);
}
```

**Why This Matters**:
- Standard gradient can cancel out when a Gaussian is under-reconstructed from multiple views
- Absolute gradient accumulates regardless of direction
- **Clone** (small Gaussians): Uses standard gradient → good for filling gaps
- **Split** (large Gaussians): Uses absolute gradient → better for breaking up over-reconstructed regions

### 11.2 FastGS Multi-View Metric Scoring Algorithm

**Algorithm Overview** (ref_impl/FastGS/utils/fast_utils.py:45-105):

```python
def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    full_metric_counts = None
    full_metric_score = None
    
    for view in range(len(camlist)):  # Default: 10 cameras
        my_viewpoint_cam = camlist[view]
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        
        # Step 1: Compute normalized L1 loss per pixel
        l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))
        
        # Step 2: Threshold to create binary metric map
        metric_map = (l1_loss_norm > args.loss_thresh).int()  # loss_thresh = 0.1
        
        # Step 3: Render again with metric mode to get per-Gaussian counts
        render_pkg = render_fastgs(..., get_flag=True, metric_map=metric_map)
        accum_loss_counts = render_pkg["accum_metric_counts"]
        
        # Step 4: Accumulate photometric score (weighted by metric counts)
        photometric_loss = 0.8 * L1 + 0.2 * (1 - SSIM)
        full_metric_score += photometric_loss * accum_loss_counts
        
        if DENSIFY:
            full_metric_counts += accum_loss_counts
    
    # Step 5: Normalize scores
    pruning_score = (full_metric_score - min) / (max - min)
    importance_score = floor(full_metric_counts / len(camlist))
    
    return importance_score, pruning_score
```

**CUDA Metric Accumulation** (ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:403-405):
```cpp
// During rendering, if this pixel is flagged in metric_map
if (metric_map[pix_id] == 1) {
    atomicAdd(&(metricCount[collected_id[j]]), 1);
}
```

**tinygs Implementation** (tinygs/src/strategy/fastgs.cu:153-200):
- Same algorithm but fully native C++/CUDA
- No Python overhead
- Direct GPU memory access without PyTorch tensor overhead
- Thrust-based parallel reductions for score computation

### 11.3 FastGS Budget-Based Pruning

**Key Innovation**: Instead of pruning all Gaussians meeting criteria, sample based on importance weight.

**Reference Implementation** (ref_impl/FastGS/scene/gaussian_model.py:505-518):
```python
scores = 1 - pruning_score  # Higher score = more important to keep
to_remove = torch.sum(prune_mask)
remove_budget = int(0.5 * to_remove)  # Only remove 50% of candidates

if remove_budget:
    padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
    padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
    
    # Multinomial sampling with importance weights
    sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
    selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool)
    selected_pts_mask[sampled_indices] = True
    
    final_prune = torch.logical_and(prune_mask, selected_pts_mask)
    self.prune_points(final_prune)
```

**tinygs Multinomial Sampling** (tinygs/src/random/multinomial.hpp):
```cpp
// Native CUDA implementation of multinomial sampling
// Used for budget-based pruning in FastGS strategy
```

### 11.4 FastGS Optimizer Scheduling

**Custom Schedule** (ref_impl/FastGS/scene/gaussian_model.py:225-244):
```python
def optimizer_step(self, iteration):
    if iteration <= 15000:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if iteration % 16 == 0:
            self.shoptimizer.step()  # Separate SH optimizer (slower)
            self.shoptimizer.zero_grad(set_to_none=True)
    elif iteration <= 20000:
        if iteration % 32 == 0:
            self.optimizer.step()
            self.shoptimizer.step()
    else:
        if iteration % 64 == 0:
            self.optimizer.step()
            self.shoptimizer.step()
```

**Key Differences from Original**:
1. **Dual Optimizers**: Separate optimizer for SH coefficients (`shoptimizer`)
2. **Reduced Frequency**: After 15k iterations, optimize less frequently
3. **Speed vs Quality Tradeoff**: Fewer optimizer steps = faster but potentially lower quality

### 11.5 AbsGS Unique Features

**1. Gaussian Weight Tracking** (ref_impl/AbsGS/train.py:119-121):
```python
gs_w = render_pkg["gs_w"]  # Accumulated alpha*T for each Gaussian
gaussians.max_weight[visibility_filter] = torch.max(
    gaussians.max_weight[visibility_filter],
    gs_w[visibility_filter])
```

**2. Opacity Reduction** (ref_impl/AbsGS/scene/gaussian_model.py:260-263):
```python
def reduce_opacity(self):
    # Reduce opacity cap to 0.8 to help remove floaters
    opacities_new = inverse_sigmoid(
        torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8))
```

**3. Initial Pruning** (ref_impl/AbsGS/scene/gaussian_model.py:360-372):
```python
def initial_prune(self):
    # Remove extremely large Gaussians at initialization
    pts_mask_1 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling)
    if len(self.get_scaling) < 5_000_000:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.quantile(self.get_scaling, 0.999)
    else:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling) * 4
```

**4. Weight-Based Pruning** (ref_impl/AbsGS/train.py:138-142):
```python
if opt.use_prune_weight:
    prune_mask = (gaussians.max_weight < opt.min_weight).squeeze()  # min_weight = 0.7
    gaussians.prune_points(prune_mask)
```

### 11.6 Original 3DGS Unique Features

**1. Depth Regularization** (ref_impl/gaussian-splatting/train.py:128-138):
```python
if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
    invDepth = render_pkg["depth"]
    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
    depth_mask = viewpoint_cam.depth_mask.cuda()
    
    Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
    loss += Ll1depth
```

**Depth Weight Schedule** (ref_impl/gaussian-splatting/arguments/__init__.py:96-97):
```python
depth_l1_weight_init = 1.0
depth_l1_weight_final = 0.01
# Exponential decay from 1.0 to 0.01 over training
```

**2. Exposure Compensation** (ref_impl/gaussian-splatting/scene/gaussian_model.py:133-176):
```python
# Per-image exposure optimization (3x4 affine transform)
self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
self._exposure = nn.Parameter(exposure.requires_grad_(True))

# Separate optimizer with own schedule
self.exposure_optimizer = torch.optim.Adam([self._exposure])
self.exposure_scheduler_args = get_expon_lr_func(
    training_args.exposure_lr_init,  # 0.01
    training_args.exposure_lr_final,  # 0.001
    ...)
```

**3. Separate SH Optimizer** (optional):
```python
# Features DC and rest have different learning rates
{'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
{'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
```

---

## 12. Hyperparameter Comparison Table

| Parameter | gaussian-splatting | FastGS | AbsGS | tinygs default |
|-----------|-------------------|--------|-------|----------------|
| `iterations` | 30,000 | 30,000 | 30,000 | 30,000 |
| `position_lr_init` | 0.00016 | 0.00016 | 0.00016 | 0.00016 |
| `position_lr_final` | 0.0000016 | 0.0000016 | 0.0000016 | 0.0000016 |
| `feature_lr` (DC) | 0.0025 | 0.0025 | 0.0025 | 0.0025 |
| `feature_lr` (rest) | 0.0025/20=0.000125 | 0.005/20 | 0.0025/20 | 0.0025/20 |
| `opacity_lr` | 0.025 | 0.025 | 0.05 | 0.05 |
| `scaling_lr` | 0.005 | 0.005 | 0.005 | 0.005 |
| `rotation_lr` | 0.001 | 0.001 | 0.001 | 0.001 |
| `percent_dense` | **0.01** | **0.001** | **0.001** | 0.01 |
| `densify_grad_threshold` | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| `densify_grad_abs_threshold` | - | **0.0012** | **0.0004** | 0.0002 |
| `loss_thresh` (metric) | - | **0.1** | - | 0.1 |
| `importance_threshold` | - | **5** | - | 5 |
| `prune_budget_ratio` | - | **0.5** | - | 0.5 |
| `final_prune_opacity` | - | **0.1** | - | 0.1 |
| `final_prune_score` | - | **0.9** | - | 0.9 |
| `opacity_reset_value` | 0.01 | 0.01 | 0.01 | 0.01 |
| `opacity_reduce_value` | - | 0.8 | 0.8 | - |
| `min_weight_threshold` | - | - | **0.7** | - |
| `depth_l1_weight_init` | **1.0** | - | - | - |
| `depth_l1_weight_final` | **0.01** | - | - | - |
| `exposure_lr_init` | **0.01** | - | - | - |
| `exposure_lr_final` | **0.001** | - | - | - |

---

## 13. Key Thresholds and Their Impact

### 13.1 percent_dense (Clone/Split Boundary)

| Implementation | Value | Impact |
|---------------|-------|--------|
| gaussian-splatting | 0.01 | More Gaussians cloned (less aggressive splitting) |
| FastGS / AbsGS | 0.001 | More Gaussians split (more aggressive splitting) |

**Why Changed**: Smaller value means more Gaussians are considered "large" and thus split instead of cloned. This leads to faster coverage of large regions.

### 13.2 Gradient Thresholds

| Threshold | Purpose | Typical Value |
|-----------|---------|---------------|
| `densify_grad_threshold` | Clone decision | 0.0002 |
| `densify_grad_abs_threshold` (FastGS) | Split decision | 0.0012 (6x higher) |
| `densify_grad_abs_threshold` (AbsGS) | Split decision | 0.0004 (2x higher) |

**Why Different**: 
- Clone threshold is lower → clone more aggressively
- Split threshold is higher → split only when really needed
- FastGS uses even higher split threshold for faster training

### 13.3 Opacity Thresholds

| Threshold | Purpose | Value |
|-----------|---------|-------|
| `min_opacity` (prune) | Remove transparent Gaussians | 0.005 |
| `opacity_reset` | Reset during training | 0.01 |
| `opacity_reduce` (AbsGS/FastGS) | Cap opacity after densify | 0.8 |
| `final_prune_opacity` (FastGS) | Aggressive final prune | 0.1 |

---

## 14. Sparse Adam Implementation

**Original 3DGS** (ref_impl/gaussian-splatting/train.py:180-186):
```python
if use_sparse_adam:
    visible = radii > 0  # Only update Gaussians visible in current view
    gaussians.optimizer.step(visible, radii.shape[0])
    gaussians.optimizer.zero_grad(set_to_none=True)
```

**Benefits**:
- Only updates Gaussians that contributed to the rendered image
- Faster iteration time when many Gaussians are not visible
- Implemented in custom CUDA rasterizer (SparseGaussianAdam class)

**tinygs Support** (tinygs/src/optim/adam.cu):
```cpp
class AdamOptimizer : public OptimizerBase {
    void step_sparse(const bool* visible, int n, cudaStream_t stream) override;
    // Only updates parameters where visible[i] == true
};
```

---

## 15. Memory Layout Comparison

### 15.1 Image Storage

| Implementation | Layout | Notes |
|---------------|--------|-------|
| Python refs | CHW linear | Standard PyTorch tensor layout |
| tinygs | CHW tiled (8×8) | Better memory coalescing for texture access |

**tinygs Tiled Layout** (tinygs/src/cuda/common_device.cuh):
```cpp
constexpr int kImageTile = 8;
constexpr int kImageTileLog2 = 3;

inline uint get_linear_index_tiled(uint y, uint x, uint width_in_tile) {
    uint tile_y = y >> kImageTileLog2;
    uint tile_x = x >> kImageTileLog2;
    uint in_tile_y = y & 7;
    uint in_tile_x = x & 7;
    return ((tile_y * width_in_tile + tile_x) << 6) | (in_tile_y << 3) | in_tile_x;
}
```

### 15.2 Gaussian Storage

| Implementation | Layout | Notes |
|---------------|--------|-------|
| Python refs | AoS (Array of Structures) | Each Gaussian is a struct |
| tinygs | SoA (Structure of Arrays) | Separate arrays for each attribute |

**tinygs SoA Layout**:
```cpp
// Separate contiguous arrays for each attribute
float3* means;      // [N] positions
float3* scales;     // [N] scale values
float4* rotations;  // [N] quaternions
float* opacities;   // [N] opacity values
float* sh0;         // [3*N] DC coefficients
float* sh1;         // [9*N] band 1
float* sh2;         // [15*N] band 2
float* sh3;         // [21*N] band 3
```

**Benefits of SoA**:
- Better memory coalescing for GPU kernels
- Efficient access to single attributes
- Enables SoA-based SH evaluation without gather operations

---

## 16. Split Algorithm Details

### 16.1 Split Position Sampling

All implementations use the same split algorithm:

```python
# Sample N new positions (default N=2) from Gaussian distribution
samples = torch.normal(mean=zeros(3), std=get_scaling[selected_pts_mask])
# Transform by rotation matrix
new_xyz = rotation @ samples + original_xyz
# Reduce scale by factor 1/(0.8*N)
new_scaling = original_scaling / (0.8 * N)
```

**Why 0.8*N**:
- N=2: Each child has scale reduced by 1.6x
- Preserves total volume approximately: 2 * (1/1.6)³ ≈ 0.49 ≈ 0.5
- Children are smaller but together cover similar region

### 16.2 tinygs Native Implementation

```cpp
// From tinygs/src/strategy/default.cu
// CUDA kernel for split position generation
__global__ void split_kernel(
    const float3* means, const float3* scales, const float4* rotations,
    float3* new_means, float3* new_scales, float4* new_rotations,
    const int* indices, int num_splits, int samples_per_split,
    curandState* rng_states) {
    // ... parallel sampling and transformation
}
```

---

## 17. Training Loop Timing

### 17.1 FastGS 100-Second Training Claim

FastGS achieves fast training through:

1. **Metric-based Densification Control**: Only densify Gaussians flagged by multi-view metric
2. **Reduced Optimizer Steps**: After 15k iterations, reduce optimization frequency
3. **Final Pruning Phase**: Aggressive pruning 15k-30k to reduce Gaussian count
4. **Separate SH Optimizer**: Update SH coefficients less frequently

### 17.2 Iteration Breakdown

| Phase | Iterations | Key Actions |
|-------|------------|-------------|
| Warmup | 0-500 | No densification |
| Densification | 500-15000 | Clone/split every 100 iters, metric-based |
| Refinement | 15000-30000 | Final prune every 3000 iters, reduced optimizer frequency |

---

## 18. API Differences

### 18.1 Python Reference API

```python
# Render function signature (gaussian-splatting)
render(viewpoint_cam, gaussians, pipe, bg, 
       scaling_modifier=1.0, 
       use_trained_exp=False, 
       separate_sh=False)

# Render function signature (FastGS)
render_fastgs(viewpoint_cam, gaussians, pipe, bg, mult,
              get_flag=False, metric_map=None)
```

### 18.2 tinygs API

```cpp
// RasterizeContext holds all parameters
struct RasterizeContext {
    // Input
    const Camera* camera;
    const GPUGaussian3d* gaussians;
    
    // Output
    ForwardOutput fwd_output;
    BackwardOutput bwd_output;
    
    // Options
    bool metric_mode = false;
    const int* metric_map = nullptr;
    int* metric_counts = nullptr;
    
    // Stream for async execution
    cudaStream_t stream;
};
```

---

## 19. Code Reference Locations

### 19.1 Key Files by Implementation

| Feature | gaussian-splatting | FastGS | AbsGS | tinygs |
|---------|-------------------|--------|-------|--------|
| **Training Loop** | train.py:43-187 | train.py:37-178 | train.py:37-191 | orchestrator.cu |
| **Densification** | gaussian_model.py:452-469 | gaussian_model.py:468-526 | gaussian_model.py:461-477 | strategy/*.cu |
| **Metric Scoring** | - | fast_utils.py:45-105 | - | strategy/fastgs.cu:153-350 |
| **Backward Pass** | backward.cu | backward.cu | backward.cu:545-550 | kernels_backward.cuh:677-692 |
| **SH Evaluation** | forward.cu:24-76 | forward.cu:24-76 | forward.cu:24-76 | kernels_forward.cuh |
| **Optimizer** | gaussian_model.py:178-200 | gaussian_model.py:192-215 | gaussian_model.py:193-212 | optim/adam.cu |

### 19.2 Line Numbers for Critical Code

**AbsGS Homodirectional Gradient**:
- CUDA: `ref_impl/AbsGS/submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/backward.cu:548-550`
- Python binding: `ref_impl/AbsGS/submodules/diff-gaussian-rasterization-abs/rasterize_points.cu:154`

**FastGS Metric Mode**:
- Forward render: `ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:403-405`
- Python API: `ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/rasterize_points.cu:62, 107-111`

**Original 3DGS Depth Regularization**:
- Training: `ref_impl/gaussian-splatting/train.py:128-138`

---

## 20. Summary of Differences

### What tinygs Implements from Each Reference

| From gaussian-splatting | From FastGS | From AbsGS |
|------------------------|-------------|------------|
| Basic densification | Multi-view metric scoring | Absolute gradient for split |
| Sparse Adam optimizer | Budget-based pruning | Separate clone/split thresholds |
| SH degree progression | Final prune phase | Opacity reduction |
| Basic pruning | Dual optimizer schedule | Weight tracking |

### What tinygs Adds Natively

| Feature | Implementation |
|---------|---------------|
| FP16 training | fastgs_ours_fp16 rasterizer |
| Multiple rasterizer backends | 3dgs_accel, fastgs_ours, fastgs_ours_fp16 |
| Multiple strategies | default, fastgs, absgs, mcmc, improved |
| Native fused SSIM | loss/fused_ssim.cu |
| NVTX profiling | All kernels |
| Tiled image layout | common_device.cuh |
| Config-driven training | JSON configs in configs/ |

### What tinygs is Missing

| Feature | Present in | Difficulty to Add |
|---------|-----------|------------------|
| Depth regularization | gaussian-splatting | Medium |
| Exposure compensation | gaussian-splatting | Medium |
| Anti-aliasing | gaussian-splatting (MIP-Splatting) | Hard |
| GUI viewer | All Python refs | Medium |
| Separate SH optimizer | FastGS | Easy |
| Per-image exposure | gaussian-splatting | Medium |

---

## 21. Additional FastGS Parameters

### 21.1 Tile Bounding Box Multiplier (`mult`)

**Purpose**: Control the tightness of tile bounding box for each Gaussian.

**Code** (ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:246):
```cpp
uint32_t tiles_count = duplicateToTilesTouched(point_image, con_o, grid, mult, 0, 0, 0, nullptr, nullptr);
```

**Default**: `mult = 0.5`

**Impact**:
- Lower value = tighter bounding box = fewer tiles touched = faster but potentially missing coverage
- Higher value = larger bounding box = more tiles touched = slower but more accurate

### 21.2 FastGS-Specific Arguments

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mult` | 0.5 | Tile bounding box multiplier |
| `highfeature_lr` | 0.005 | Learning rate for high-frequency SH coefficients |
| `lowfeature_lr` | 0.0025 | Learning rate for DC coefficients |
| `grad_thresh` | 0.0002 | Clone gradient threshold |
| `dense` | 0.001 | Scale boundary for clone/split (percent_dense) |

---

## 22. Additional AbsGS Parameters

### 22.1 Opacity Reduction

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `use_reduce` | True | Enable opacity reduction for floater removal |
| `opacity_reduce_interval` | 500 | How often to reduce opacity |

**Code** (ref_impl/AbsGS/scene/gaussian_model.py:260-263):
```python
def reduce_opacity(self):
    # Cap opacity at 0.8 to help identify floaters
    opacities_new = inverse_sigmoid(
        torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8))
```

### 22.2 Weight-Based Pruning

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `use_prune_weight` | False | Enable weight-based pruning |
| `prune_until_iter` | 25000 | Stop weight pruning after this iteration |
| `min_weight` | 0.7 | Minimum accumulated weight threshold |

### 22.3 Initial Pruning

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `init_prune` | False | Enable initial pruning of large Gaussians |

**Code** (ref_impl/AbsGS/scene/gaussian_model.py:360-372):
```python
def initial_prune(self):
    # Remove extremely large Gaussians at initialization
    pts_mask_1 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling)
    if len(self.get_scaling) < 5_000_000:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.quantile(self.get_scaling, 0.999)
    else:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling) * 4
    selected_pts_mask = torch.logical_and(pts_mask_1, pts_mask_2)
    self.prune_points(selected_pts_mask)
```

---

## 23. Rasterizer Implementation Details

### 23.1 Forward Pass Pipeline

All implementations follow similar forward pass stages:

```
1. Preprocess: Project 3D → 2D, compute conic, bounds
2. Duplicate: Create (tile_id, depth) pairs for overlapping tiles
3. Sort: Radix sort by depth (CUB)
4. Sort: Radix sort by tile ID (CUB)
5. Identify ranges: Find start/end indices per tile
6. Render: Alpha blending per tile in parallel
```

### 23.2 Key Differences

| Aspect | Original 3DGS | FastGS | tinygs fastgs_ours |
|--------|--------------|--------|-------------------|
| Tile size | 16×16 | 16×16 | 16×16 |
| Sort backend | CUB | CUB | CUB |
| Bucket-based blending | No | No | Yes |
| Metric mode | No | Yes | Yes |
| FP16 support | No | No | Yes |
| Dual stream | No | No | Yes |
| Tiled image layout | No | No | Yes |

### 23.3 Bucket-Based Blending (tinygs Innovation)

**Purpose**: Reduce memory usage by processing Gaussians in buckets rather than all at once.

**Implementation**: Each tile's Gaussians are processed in 32-Gaussian buckets, storing intermediate transmittance values for backward pass.

---

## 24. Backward Pass Implementation

### 24.1 Gradient Computation Order

1. **Blend backward**: Propagate pixel gradients to colors and opacity
2. **Preprocess backward**: Propagate 2D gradients to 3D parameters
3. **SH backward**: Propagate color gradients to SH coefficients

### 24.2 Absolute Gradient Accumulation

**Reference** (AbsGS backward.cu:548-550):
```cpp
// Standard gradient
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

// Homodirectional (absolute) gradient
atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * ddelx_dx));
atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * ddely_dy));
```

**tinygs** (kernels_backward.cuh:677, 691-692):
```cpp
// Accumulate per-Gaussian absolute gradient
absdL_dmean2d_accum += make_float2(fabsf(dL_dmean2d.x), fabsf(dL_dmean2d.y));

// Atomic add to global buffer
if (absgrad_mean2d != nullptr) {
    atomicAdd(&absgrad_mean2d[primitive_idx].x, absdL_dmean2d_accum.x);
    atomicAdd(&absgrad_mean2d[primitive_idx].y, absdL_dmean2d_accum.y);
}
```

---

## 25. SSIM Implementation Details

### 25.1 Constants

All implementations use the same SSIM constants:
```cpp
constexpr float C1 = 0.01f * 0.01f;  // (K1 * L)^2 where K1=0.01, L=1
constexpr float C2 = 0.03f * 0.03f;  // (K2 * L)^2 where K2=0.03, L=1
```

### 25.2 Gaussian Window

- **Window size**: 11×11
- **Sigma**: 1.5
- **Separable**: Implemented as two 1D convolutions (horizontal then vertical)

### 25.3 tinygs Fused SSIM Optimizations

1. **Shared memory tiling**: 16×16 blocks with 5-pixel halo
2. **Half2 vectorization**: Load two pixels at once in FP16 variant
3. **Warp-level reduction**: Efficient partial sum reduction
4. **Tiled image layout**: Memory coalescing optimization
5. **Backward pass**: Stores partial derivatives for efficient gradient computation

---

## 26. Performance Considerations

### 26.1 Memory Access Patterns

| Implementation | Image Access | Gaussian Access |
|---------------|--------------|-----------------|
| Python refs | Linear CHW | AoS |
| tinygs | Tiled CHW (8×8) | SoA |

### 26.2 Kernel Launch Configuration

| Kernel | Block Size | Grid Strategy |
|--------|-----------|---------------|
| Preprocess | 256 | One thread per Gaussian |
| Render | 16×16 | One block per tile |
| Backward | 16×16 | One block per tile |
| SSIM | 16×16 | One block per output region |

### 26.3 Stream Usage

**tinygs**: Dual stream execution
- Stream 1: Forward pass
- Stream 2: Loss computation, backward pass preparation

**Python refs**: Single stream (implicit PyTorch default)

---

## 27. Debugging and Profiling

### 27.1 NVTX Ranges (tinygs)

```cpp
NVTX3_FUNC_RANGE();  // Every major function
NVTX3_RANGE("custom_name");  // Custom ranges
```

### 27.2 Validation Checks

| Check | Location | Purpose |
|-------|----------|---------|
| NaN gradient | After backward | Detect numerical instability |
| Bounds check | Kernel entry | Prevent out-of-bounds access |
| Opacity range | After activation | Ensure [0,1] range |

### 27.3 Common Issues

1. **NaN gradients**: Usually from zero division in gradient normalization
2. **Memory overflow**: Too many Gaussians in densification phase
3. **Slow convergence**: Learning rate schedule mismatch

---

## 28. Precise Code Reference Locations

### 28.1 AbsGS Homodirectional Gradient (Complete Implementation)

**Step 1: Backward Pass - Compute Absolute Gradient**

File: `ref_impl/AbsGS/submodules/diff-gaussian-rasterization-abs/cuda_rasterizer/backward.cu:544-550`
```cpp
// Standard gradient accumulation (for clone decisions)
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

// Homodirectional gradient accumulation (for split decisions)
// KEY DIFFERENCE: Use fabs() instead of signed value
atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * ddelx_dx));
atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * ddely_dy));
```

**Step 2: Python Binding - Allocate float4 Instead of float2**

File: `ref_impl/AbsGS/submodules/diff-gaussian-rasterization-abs/rasterize_points.cu:154`
```cpp
// Changed from float2 to float4 to store both gradients
torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
```

**Step 3: Gradient Accumulation in Python**

File: `ref_impl/AbsGS/scene/gaussian_model.py:479-482`
```python
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    # Standard gradient: L2 norm of first 2 components (indices 0:2)
    self.xyz_gradient_accum[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    # Absolute gradient: L2 norm of last 2 components (indices 2:4)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
    self.denom[update_filter] += 1
```

**tinygs Implementation**

File: `tinygs/src/rasterizer/fastgs_ours/kernels_backward.cuh:677, 690-692`
```cpp
// Accumulate absolute gradient during backward pass
absdL_dmean2d_accum += make_float2(fabsf(dL_dmean2d.x), fabsf(dL_dmean2d.y));
// ...
if (absgrad_mean2d != nullptr) {
    atomicAdd(&absgrad_mean2d[primitive_idx].x, absdL_dmean2d_accum.x);
    atomicAdd(&absgrad_mean2d[primitive_idx].y, absdL_dmean2d_accum.y);
}
```

### 28.2 FastGS Metric Mode (Complete Implementation)

**Step 1: Forward Render with Metric Counting**

File: `ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:401-407`
```cpp
// During alpha blending, check if this pixel is flagged in metric_map
if(get_flag) {
    if(metric_map[pix_id] == 1) {
        atomicAdd(&(metricCount[collected_id[j]]), 1);
    }
}
```

**Step 2: Python API for Metric Render**

File: `ref_impl/FastGS/gaussian_renderer/__init__.py:37-55`
```python
# Create metric_map if not provided
if metric_map==None:
    metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), 
                           dtype=torch.int, device='cuda')

raster_settings = GaussianRasterizationSettings(
    # ... other params ...
    mult = mult,
    get_flag=get_flag,        # Enable metric counting
    metric_map = metric_map   # Binary mask of high-loss pixels
)
```

**Step 3: Multi-View Score Computation**

File: `ref_impl/FastGS/utils/fast_utils.py:45-105`
```python
def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    for view in range(len(camlist)):  # Default: 10 cameras
        # First render: get rendered image
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        
        # Compute normalized L1 loss per pixel
        l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))
        
        # Threshold to binary metric map
        metric_map = (l1_loss_norm > args.loss_thresh).int()  # loss_thresh = 0.1
        
        # Second render with metric mode
        render_pkg = render_fastgs(..., get_flag=True, metric_map=metric_map)
        accum_loss_counts = render_pkg["accum_metric_counts"]
        
        # Accumulate scores
        full_metric_score += photometric_loss * accum_loss_counts
        if DENSIFY:
            full_metric_counts += accum_loss_counts
    
    # Normalize
    pruning_score = (full_metric_score - min) / (max - min)
    importance_score = floor(full_metric_counts / num_cameras)
    return importance_score, pruning_score
```

### 28.3 Gaussian Weight Output (AbsGS Unique Feature)

**Purpose**: Track accumulated alpha × transmittance for each Gaussian to identify floaters.

File: `ref_impl/AbsGS/submodules/diff-gaussian-rasterization-abs/rasterize_points.cu:70, 112, 116`
```cpp
// Allocate weight buffer
torch::Tensor gs_w = torch::full({P}, 0.0, means3D.options());

// Forward pass fills gs_w
rendered = CudaRasterizer::Rasterizer::forward(
    // ... params ...
    gs_w.contiguous().data<float>(),  // Output: per-Gaussian weight
    radii.contiguous().data<int>(),
    debug);

return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, gs_w);
```

File: `ref_impl/AbsGS/train.py:94, 119-121`
```python
# During training, extract and track weights
render_pkg = render(viewpoint_cam, gaussians, pipe, background)
gs_w = render_pkg["gs_w"]  # Accumulated alpha*T for each Gaussian

# Track max weight for floater identification
gaussians.max_weight[visibility_filter] = torch.max(
    gaussians.max_weight[visibility_filter],
    gs_w[visibility_filter])
```

File: `ref_impl/AbsGS/train.py:138-142`
```python
# Weight-based pruning (optional)
if opt.use_prune_weight:
    prune_mask = (gaussians.max_weight < opt.min_weight).squeeze()  # min_weight = 0.7
    gaussians.prune_points(prune_mask)
```

---

## 29. AbsGS Unique Features Not in Other Implementations

### 29.1 Initial Pruning (Remove Large Gaussians at Initialization)

File: `ref_impl/AbsGS/scene/gaussian_model.py:360-372`
```python
def initial_prune(self):
    """Remove extremely large Gaussians that are likely initialization noise."""
    # Criterion 1: Scale larger than mean
    pts_mask_1 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling)
    
    # Criterion 2: Scale in top 0.1% or > 4× mean
    if len(self.get_scaling) < 5_000_000:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.quantile(
            self.get_scaling, 0.999)
    else:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.mean(
            self.get_scaling) * 4
    
    # Remove Gaussians matching both criteria
    selected_pts_mask = torch.logical_and(pts_mask_1, pts_mask_2)
    self.prune_points(selected_pts_mask)
```

**When Enabled**: `init_prune = True` in ModelParams (default: False)

### 29.2 Opacity Reduction (Floater Prevention)

File: `ref_impl/AbsGS/scene/gaussian_model.py:260-263`
```python
def reduce_opacity(self):
    """Cap opacity at 0.8 to help identify and remove floaters.
    
    Gaussians that are true geometry will be reinforced by multiple views,
    while floaters (visible from few views) will have their opacity reduced
    and eventually pruned.
    """
    opacities_new = inverse_sigmoid(
        torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.8))
    self._opacity = opacities_new
```

**Schedule**: Every 500 iterations during densification (use_reduce=True, opacity_reduce_interval=500)

### 29.3 Weight-Based Pruning Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_prune_weight` | False | Enable weight-based floater pruning |
| `prune_until_iter` | 25000 | Stop weight pruning after this iteration |
| `min_weight` | 0.7 | Minimum accumulated weight threshold |

---

## 30. FastGS Optimizer Scheduling Details

### 30.1 Dual Optimizer Architecture

File: `ref_impl/FastGS/scene/gaussian_model.py:198-212`
```python
def training_setup(self, training_args):
    # Main optimizer: position, DC features, opacity, scale, rotation
    l = [
        {'params': [self._xyz], 'lr': position_lr, "name": "xyz"},
        {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
        {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    ]
    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    # Separate SH optimizer: high-frequency SH coefficients (updated less frequently)
    sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]
    self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
```

### 30.2 Optimizer Step Schedule

File: `ref_impl/FastGS/scene/gaussian_model.py:225-244`
```python
def optimizer_step(self, iteration):
    """Reduce optimization frequency after 15k iterations for speed."""
    if iteration <= 15000:
        # Phase 1: Normal optimization
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if iteration % 16 == 0:  # SH updated every 16 iterations
            self.shoptimizer.step()
            self.shoptimizer.zero_grad(set_to_none=True)
    elif iteration <= 20000:
        # Phase 2: Reduce main optimizer to every 32 iterations
        if iteration % 32 == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.shoptimizer.step()
            self.shoptimizer.zero_grad(set_to_none=True)
    else:
        # Phase 3: Reduce to every 64 iterations (final refinement)
        if iteration % 64 == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.shoptimizer.step()
            self.shoptimizer.zero_grad(set_to_none=True)
```

### 30.3 Learning Rate Configuration

| Parameter | FastGS | Original 3DGS |
|-----------|--------|---------------|
| `feature_lr` (DC) | 0.0025 | 0.0025 |
| `highfeature_lr` | 0.005 | N/A (single optimizer) |
| `lowfeature_lr` | 0.0025 | N/A (single optimizer) |
| `opacity_lr` | 0.025 | 0.025 |

---

## 31. tinygs Implementation Specifics

### 31.1 Multinomial Sampling for Budget-Based Pruning

File: `tinygs/src/random/multinomial.hpp` (used by FastGS strategy)

```cpp
// Native CUDA implementation of weighted sampling without replacement
// Used when budget < number of prune candidates
GPUBuffer<int> multinomial_cuda_cpu_without_replacement(
    const float* weights,    // Sampling weights (higher = more likely)
    int num_candidates,      // Number of candidates
    int num_samples,         // Number to sample (budget)
    int seed,                // RNG seed
    cudaStream_t stream);
```

File: `tinygs/src/strategy/fastgs.cu:828-849`
```cpp
// Compute prune weights: higher pruning_score → higher weight → more likely pruned
thrust::transform(exec,
    candidate_indices.begin(), candidate_indices.end(),
    prune_weights.begin(),
    [d_ps] __device__(int i) -> float {
        float score = d_ps[i];
        if (!isfinite(score)) score = 0.0f;
        score = fminf(fmaxf(score, 0.0f), 1.0f);
        return 1.0f / (1e-6f + 1.0f - score);  // Inverse: high score = high weight
    });

// Sample without replacement
GPUBuffer<int> sampled_positions = multinomial_cuda_cpu_without_replacement(
    thrust::raw_pointer_cast(prune_weights.data()),
    num_standard_candidates,
    budget,  // prune_budget_ratio * num_candidates
    seed,
    ctx.stream);
```

### 31.2 Absolute Gradient Accumulation Pattern

File: `tinygs/src/strategy/fastgs.cu:533-559`
```cpp
// Classify each Gaussian for densification
thrust::for_each(exec,
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gaussians),
    [d_densification_info, d_scale, d_grow_flags, d_importance,
     clone_thresh, split_thresh, scale_boundary, importance_thresh] __device__(int i) {
        const float counter = fmaxf(d_densification_info[i].accum_counter, 1.0f);
        float grad = d_densification_info[i].accum_grad_mean2d / counter;
        float absgrad = d_densification_info[i].accum_absgrad_mean2d / counter;
        
        const float max_scale = max(activate_scale(d_scale[i]));
        bool importance_ok = (d_importance == nullptr) || (d_importance[i] > importance_thresh);
        
        if (counter > 0 && importance_ok) {
            // Clone: standard gradient + small scale
            if (grad >= clone_thresh && max_scale <= scale_boundary) {
                d_grow_flags[i] = kClone;
            }
            // Split: absolute gradient + large scale
            else if (absgrad >= split_thresh && max_scale > scale_boundary) {
                d_grow_flags[i] = kSplit;
            }
        }
    });
```

### 31.3 DensificationInfo Structure

File: `tinygs/src/strategy/base.hpp`
```cpp
struct DensificationInfo {
    float accum_grad_mean2d;      // Accumulated standard gradient (for clone)
    float accum_absgrad_mean2d;   // Accumulated absolute gradient (for split)
    float accum_counter;          // Number of times this Gaussian was visible
    float max_radii_screen;       // Maximum screen-space radius
    float metric_importance_score; // FastGS multi-view importance
    float metric_pruning_score;    // FastGS pruning priority
};
```

---

## 32. Training Loop Timeline Comparison

### 32.1 Original 3DGS Timeline

```
Iter 0-500:      Warmup (no densification)
Iter 500-15000:  Densify every 100 iters
                 - Clone if grad >= thresh AND scale <= percent_dense*extent
                 - Split if grad >= thresh AND scale > percent_dense*extent
                 - Prune if opacity < 0.005 OR too large
                 - Reset opacity every 3000
Iter 15000-30000: Refinement only (no densification)
```

### 32.2 FastGS Timeline

```
Iter 0-500:      Warmup (no densification)
Iter 500-15000:  Densify every 100 iters with multi-view metric
                 - Compute importance/pruning scores from 10 random views
                 - Clone if grad >= thresh AND importance > 5 AND small scale
                 - Split if absgrad >= thresh AND importance > 5 AND large scale
                 - Budget-based prune (only 50% of candidates)
                 - Opacity cap at 0.8
Iter 15000-30000: Final prune every 3000
                 - Prune if opacity < 0.1 OR pruning_score > 0.9
                 
Optimizer schedule:
  0-15000:       Main every iter, SH every 16
  15000-20000:   Both every 32
  20000-30000:   Both every 64
```

### 32.3 AbsGS Timeline

```
Iter 0:          Optional initial prune of large Gaussians
Iter 0-500:      Warmup (no densification)
Iter 500-15000:  Densify every 100 iters with separate gradients
                 - Clone if norm(grad) >= clone_thresh AND small scale
                 - Split if norm(absgrad) >= absgrad_thresh AND large scale
                 - Prune if opacity < 0.005 OR too large
                 - Opacity reduce every 500 (cap at 0.8)
                 - Reset opacity every 3000
Iter 500-25000:  Optional weight-based prune (if use_prune_weight)
                 - Prune if max_weight < 0.7
```

---

## 33. File Reference Summary

### 33.1 gaussian-splatting Key Files

| File | Lines of Interest | Purpose |
|------|------------------|---------|
| `train.py` | 128-138 | Depth regularization |
| `train.py` | 178-186 | Exposure optimizer + sparse Adam |
| `arguments/__init__.py` | 85-87 | Exposure learning rates |
| `arguments/__init__.py` | 96-97 | Depth weight schedule |
| `scene/gaussian_model.py` | 133-176 | Exposure compensation parameters |

### 33.2 FastGS Key Files

| File | Lines of Interest | Purpose |
|------|------------------|---------|
| `train.py` | 132-148 | Metric-based densification |
| `train.py` | 150-158 | Final prune phase |
| `train.py` | 162-167 | Sparse Adam handling |
| `arguments/__init__.py` | 94-101 | FastGS-specific hyperparameters |
| `scene/gaussian_model.py` | 225-244 | Optimizer schedule |
| `scene/gaussian_model.py` | 468-527 | densify_and_prune_fastgs |
| `scene/gaussian_model.py` | 533-540 | final_prune_fastgs |
| `utils/fast_utils.py` | 45-105 | compute_gaussian_score_fastgs |
| `gaussian_renderer/__init__.py` | 37-55 | Metric map integration |
| `cuda_rasterizer/forward.cu` | 401-407 | Metric counting kernel |

### 33.3 AbsGS Key Files

| File | Lines of Interest | Purpose |
|------|------------------|---------|
| `train.py` | 37-38 | Initial prune call |
| `train.py` | 94, 119-121 | Weight tracking |
| `train.py` | 128-142 | Densification with absgrad |
| `arguments/__init__.py` | 57 | init_prune flag |
| `arguments/__init__.py` | 90 | densify_grad_abs_threshold |
| `arguments/__init__.py` | 92-97 | Opacity reduce and weight prune params |
| `scene/gaussian_model.py` | 260-263 | reduce_opacity |
| `scene/gaussian_model.py` | 360-372 | initial_prune |
| `scene/gaussian_model.py` | 461-477 | densify_and_prune with absgrad |
| `scene/gaussian_model.py` | 479-482 | add_densification_stats (separate gradients) |
| `cuda_rasterizer/backward.cu` | 544-550 | Homodirectional gradient computation |
| `rasterize_points.cu` | 70, 112, 116 | gs_w output |
| `rasterize_points.cu` | 154 | float4 for dL_dmean2D |

### 33.4 tinygs Key Files

| File | Lines of Interest | Purpose |
|------|------------------|---------|
| `strategy/default.cu` | 40-68 | step_impl (original strategy) |
| `strategy/absgs.cu` | 37-66 | step_impl (AbsGS strategy) |
| `strategy/absgs.cu` | 99-117 | Separate clone/split decisions |
| `strategy/fastgs.cu` | 153-430 | compute_gaussian_score |
| `strategy/fastgs.cu` | 502-752 | duplicate (clone+split) |
| `strategy/fastgs.cu` | 759-914 | prune (budget-based) |
| `strategy/fastgs.cu` | 921-955 | final_prune |
| `rasterizer/fastgs_ours/kernels_backward.cuh` | 677, 690-692 | Absolute gradient accumulation |

---

## 34. Fused SSIM Implementation Comparison

### 34.1 FastGS Fused SSIM (ref_impl/FastGS/submodules/fused-ssim/ssim.cu)

**Key Implementation Details**:
- Block size: 32×32 pixels
- Gaussian kernel: 11-tap 1D separable (horizontal then vertical)
- Shared memory: 42×42 per block (32 + 10 halo pixels each side)
- Pre-computed Gaussian weights (G_00 to G_10) hardcoded as constants

**Forward Pass**:
```cpp
// 11-tap Gaussian weights (sigma ≈ 1.5)
#define G_00 0.001028380123898387f
#define G_01 0.0075987582094967365f
#define G_02 0.036000773310661316f
#define G_03 0.10936068743467331f
#define G_04 0.21300552785396576f
#define G_05 0.26601171493530273f  // Center
// ... symmetric weights G_06 to G_10

// Forward kernel computes:
// mu1, mu2 (local means)
// sigma1_sq, sigma2_sq (local variances)
// sigma12 (local covariance)
// SSIM = (2*mu1*mu2 + C1)(2*sigma12 + C2) / ((mu1²+mu2²+C1)(sigma1²+sigma2²+C2))

// Stores partial derivatives for backward pass:
dm_dmu1[global_idx] = derivative of SSIM w.r.t. mu1
dm_dsigma1_sq[global_idx] = derivative of SSIM w.r.t. sigma1_sq
dm_dsigma12[global_idx] = derivative of SSIM w.r.t. sigma12
```

**Backward Pass** (lines 287-365):
```cpp
// Backpropagate through SSIM using stored partial derivatives
// dL/dimg1 = dL/dSSIM * (dSSIM/dmu1 * dmu1/dimg1 + dSSIM/dsigma1_sq * dsigma1_sq/dimg1 + dSSIM/dsigma12 * dsigma12/dimg1)
```

### 34.2 tinygs Fused SSIM (tinygs/src/loss/fused_ssim.cu)

**Key Differences**:
- Block size: 16×16 pixels
- Tiled image layout support (8×8 tiles)
- FP32 and FP16 variants
- Two-pass separable convolution similar to FastGS

**Architecture**:
```cpp
// FP32 forward kernel
__global__ void fused_ssim_cuda_fp32(
    int H, int W, float C1, float C2, float scale,
    const float* img1, const float* img2,
    float* ssim_map,
    float* dm_dmu1,      // Partial derivatives for backward
    float* dm_dsigma1_sq,
    float* dm_dsigma12
);

// FP16 variant for memory efficiency
__global__ void fused_ssim_cuda_fp16(...);
```

### 34.3 SSIM Constants Comparison

| Constant | Value | Derivation |
|----------|-------|------------|
| C1 | 0.0001 | (K1 * L)² = (0.01 * 1)² |
| C2 | 0.0009 | (K2 * L)² = (0.03 * 1)² |
| Window size | 11×11 | Standard SSIM window |
| Sigma | ~1.5 | Gaussian blur sigma |

---

## 35. Sparse Adam CUDA Implementation

### 35.1 FastGS Sparse Adam (ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/adam.cu)

**Key Implementation**:
```cpp
__global__ void adamUpdateCUDA(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,      // First moment
    float* __restrict__ exp_avg_sq,   // Second moment
    const bool* tiles_touched,        // Visibility mask
    const float lr,
    const float b1,                   // Beta1 (default 0.9)
    const float b2,                   // Beta2 (default 0.999)
    const float eps,                  // Epsilon (default 1e-15)
    const uint32_t N,                 // Number of Gaussians
    const uint32_t M) {               // Parameters per Gaussian (e.g., 3 for xyz)

    auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;  // Gaussian index
    if (g_idx >= N) return;

    // Only update visible Gaussians
    if (tiles_touched[g_idx]) {
        float grad = param_grad[p_idx];
        float m = exp_avg[p_idx];
        float v = exp_avg_sq[p_idx];
        
        // Adam update
        m = b1 * m + (1.0f - b1) * grad;
        v = b2 * v + (1.0f - b2) * grad * grad;
        float step = -lr * m / (sqrtf(v) + eps);
        
        param[p_idx] += step;
        exp_avg[p_idx] = m;
        exp_avg_sq[p_idx] = v;
    }
}
```

### 35.2 Sparse Adam Benefits

| Aspect | Dense Adam | Sparse Adam |
|--------|------------|-------------|
| Update scope | All Gaussians | Only visible Gaussians |
| Memory per step | O(N) | O(visible) |
| Speed (large scenes) | Slower | Faster |
| Quality | Same | Same |

---

## 36. Detailed Backward Pass Gradient Flow

### 36.1 Original 3DGS Backward (ref_impl/gaussian-splatting)

**Gradient Flow**:
```
dL/dpixels (from loss)
    ↓
renderCUDA (backward blending)
    → dL/dcolors (per-Gaussian color gradients)
    → dL/dopacity (per-Gaussian opacity gradients)
    → dL/dmean2D (float2: per-Gaussian 2D position gradients)
    → dL/dconic2D (per-Gaussian conic gradients)
    ↓
preprocessCUDA (backward projection)
    → dL/dmeans3D (3D position gradients)
    → dL/dscales (scale gradients)
    → dL/drotations (rotation gradients)
    → dL/dsh (SH coefficient gradients)
```

### 36.2 AbsGS Backward (with Homodirectional Gradient)

**Modified Gradient Flow**:
```
dL/dpixels
    ↓
renderCUDA
    → dL/dcolors
    → dL/dopacity
    → dL/dmean2D (float4: stores BOTH standard AND absolute gradients!)
        [0]: dL/dx (signed)
        [1]: dL/dy (signed)
        [2]: |dL/dx| (absolute) ← NEW!
        [3]: |dL/dy| (absolute) ← NEW!
    → dL/dconic2D
    ↓
preprocessCUDA
    → dL/dmeans3D
    → dL/dscales
    → dL/drotations
    → dL/dsh
```

**Critical Code** (backward.cu:544-550):
```cpp
// Standard gradient accumulation (for clone decisions)
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

// Homodirectional gradient accumulation (for split decisions)
// KEY: Use fabs() instead of signed value
atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * ddelx_dx));
atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * ddely_dy));
```

### 36.3 Why Absolute Gradients Help Splitting

**Problem with Signed Gradients**:
- A Gaussian under-reconstructed from multiple views gets gradients pointing in different directions
- These gradients can cancel out, making the Gaussian appear "good"
- The Gaussian doesn't get densified, perpetuating the artifact

**Solution with Absolute Gradients**:
- Absolute gradients accumulate regardless of direction
- A large Gaussian receiving many gradients from different views will have high abs gradient
- This correctly triggers splitting to break up the over-reconstructed region

---

## 37. Metric Mode Implementation Details

### 37.1 FastGS Metric Mode (forward.cu:401-407)

**During Rendering**:
```cpp
// In renderCUDA kernel, after computing alpha blending:
if (get_flag) {
    if (metric_map[pix_id] == 1) {
        // This pixel has high error - accumulate to all Gaussians touching it
        atomicAdd(&(metricCount[collected_id[j]]), 1);
    }
}
```

**Python Usage** (fast_utils.py:73-105):
```python
for view in range(len(camlist)):  # 10 cameras
    # First render: get rendered image
    render_image = render_fastgs(cam, gaussians, pipe, bg, mult)["render"]
    
    # Compute normalized L1 loss per pixel
    l1_loss = torch.mean(torch.abs(render_image - gt_image), 0)
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))
    
    # Threshold to binary metric map
    metric_map = (l1_loss_norm > args.loss_thresh).int()  # Default: 0.1
    
    # Second render with metric counting
    render_pkg = render_fastgs(cam, gaussians, pipe, bg, mult, 
                               get_flag=True, metric_map=metric_map)
    accum_metric_counts = render_pkg["accum_metric_counts"]
    
    # Accumulate scores
    photometric_loss = 0.8 * L1 + 0.2 * (1 - SSIM)
    full_metric_score += photometric_loss * accum_metric_counts
    full_metric_counts += accum_metric_counts

# Normalize
pruning_score = (full_metric_score - min) / (max - min)
importance_score = floor(full_metric_counts / num_cameras)
```

### 37.2 tinygs Metric Mode (fastgs.cu:153-350)

**Same algorithm but native C++/CUDA**:
- Uses `thrust::device_vector` for accumulators
- Direct GPU memory access without PyTorch overhead
- Configurable number of cameras (default: 10)
- Supports both with/without replacement sampling

---

## 38. Gaussian Weight Tracking (AbsGS Unique Feature)

### 38.1 Purpose

Track accumulated α × T (alpha times transmittance) for each Gaussian across all training views to identify floaters.

### 38.2 Implementation

**Forward Pass Output** (rasterize_points.cu:70):
```cpp
torch::Tensor gs_w = torch::full({P}, 0.0, means3D.options());
```

**Accumulation** (train.py:119-121):
```python
render_pkg = render(viewpoint_cam, gaussians, pipe, background)
gs_w = render_pkg["gs_w"]  # Per-Gaussian weight from forward

# Track max weight
gaussians.max_weight[visibility_filter] = torch.max(
    gaussians.max_weight[visibility_filter],
    gs_w[visibility_filter])
```

**Pruning** (train.py:138-142):
```python
if opt.use_prune_weight:
    # Gaussians with low max_weight are floaters
    prune_mask = (gaussians.max_weight < opt.min_weight).squeeze()  # Default: 0.7
    gaussians.prune_points(prune_mask)
```

### 38.3 Interpretation

- High `max_weight`: Gaussian is consistently visible and contributes significantly → true geometry
- Low `max_weight`: Gaussian is barely visible or has low contribution → likely a floater

---

## 39. Depth Regularization (Original 3DGS Only)

### 39.1 Implementation (train.py:128-138)

```python
if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
    invDepth = render_pkg["depth"]
    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
    depth_mask = viewpoint_cam.depth_mask.cuda()
    
    Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
    loss += Ll1depth
```

### 39.2 Depth Weight Schedule

```python
# From arguments/__init__.py:96-97
depth_l1_weight_init = 1.0
depth_l1_weight_final = 0.01

# Exponential decay from 1.0 to 0.01 over 30,000 iterations
# High early weight helps geometry, low late weight preserves details
```

---

## 40. Exposure Compensation (Original 3DGS Only)

### 40.1 Implementation

**Per-Image Exposure Parameters** (gaussian_model.py:133-176):
```python
# 3×4 affine transform per image
self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
self._exposure = nn.Parameter(exposure.requires_grad_(True))

# Separate optimizer
self.exposure_optimizer = torch.optim.Adam([self._exposure])
```

**Learning Rate Schedule**:
```python
exposure_lr_init = 0.01
exposure_lr_final = 0.001
```

### 40.2 Purpose

- Compensate for varying exposure across training images
- Each image learns its own 3×4 affine color transform
- Improves results for datasets with inconsistent camera settings

---

## 41. Training Differences Summary Table

| Feature | gaussian-splatting | FastGS | AbsGS | tinygs |
|---------|-------------------|--------|-------|--------|
| **Loss Function** | L1 + SSIM | L1 + fused SSIM | L1 + SSIM | L1 + fused SSIM |
| **Depth Regularization** | ✅ | ❌ | ❌ | ❌ |
| **Exposure Compensation** | ✅ | ❌ | ❌ | ❌ |
| **Separate SH Optimizer** | Optional | ✅ (dual) | ❌ | ❌ |
| **Sparse Adam** | ✅ | ✅ | ❌ | ✅ |
| **Absolute Gradient** | ❌ | ✅ | ✅ | ✅ |
| **Multi-view Metric** | ❌ | ✅ | ❌ | ✅ |
| **Budget-based Pruning** | ❌ | ✅ | ❌ | ✅ |
| **Opacity Reduction** | ❌ | ✅ (0.8) | ✅ (0.8) | ✅ |
| **Initial Pruning** | ❌ | ❌ | ✅ | ❌ |
| **Weight-based Pruning** | ❌ | ❌ | ✅ | ❌ |
| **Final Aggressive Prune** | ❌ | ✅ | ❌ | ✅ |
| **FP16 Training** | ❌ | ❌ | ❌ | ✅ |
| **NVTX Profiling** | ❌ | ❌ | ❌ | ✅ |

---

## 42. Code Architecture Comparison

### 42.1 Python Reference Architecture

```
Training Loop (train.py)
    ├── Scene (scene/__init__.py)
    │   └── GaussianModel (scene/gaussian_model.py)
    │       └── Optimizer state management
    ├── GaussianRenderer (gaussian_renderer/__init__.py)
    │   └── CUDA Rasterizer (submodules/diff-gaussian-rasterization/)
    │       ├── forward.cu: Preprocess + Render
    │       └── backward.cu: Gradient propagation
    └── Utils (utils/)
        ├── loss_utils.py: L1, SSIM
        └── general_utils.py: Learning rate schedules
```

### 42.2 tinygs Architecture

```
Orchestrator (orchestrator.cu)
    ├── GPUGaussian3d (core/gpu_gaussian.hpp)
    │   └── SoA parameter storage
    ├── RasterizerBase (rasterizer/rasterizer.hpp)
    │   ├── fastgs_ours: Custom implementation
    │   ├── fastgs_ours_fp16: FP16 variant
    │   └── 3dgs_accel: Original algorithm
    ├── StrategyBase (strategy/strategy.hpp)
    │   ├── DefaultStrategy: Original 3DGS
    │   ├── FastGSStrategy: Multi-view metric
    │   ├── AbsGSStrategy: Absolute gradient
    │   ├── MCMCStrategy: Markov Chain Monte Carlo
    │   └── ImprovedStrategy: Enhanced default
    ├── OptimizerBase (optim/optim.hpp)
    │   ├── AdamOptimizer
    │   ├── AdamWOptimizer
    │   └── SGDOptimizer
    └── Loss Functions (loss/)
        ├── L1Loss
        └── FusedSSIMLoss
```

---

## 43. Performance Optimization Techniques

### 43.1 tinygs Optimizations

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Tiled Image Layout** | 8×8 tiles for memory coalescing | ~10-20% faster texture access |
| **Dual Stream Execution** | Forward + Loss/Backward overlap | Better GPU utilization |
| **FP16 Support** | Half-precision training | 2× memory reduction, ~1.5× speedup |
| **NVTX Ranges** | Profiling integration | Easy performance debugging |
| **Bucket-based Blending** | Process 32 Gaussians per bucket | Reduced memory for backward pass |
| **SoA Layout** | Structure of Arrays | Better cache utilization |

### 43.2 FastGS Speed Optimizations

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Metric-based Densification** | Only densify flagged Gaussians | Fewer total Gaussians |
| **Reduced Optimizer Steps** | Every 32/64 iterations after 15k | ~2× speedup in refinement |
| **Dual Optimizer** | SH updated less frequently | ~10% speedup |
| **Fused SSIM** | Single kernel for SSIM | ~5-10% speedup |

---

## 44. Hyperparameter Defaults Comparison

### 44.1 Learning Rates

| Parameter | gaussian-splatting | FastGS | AbsGS |
|-----------|-------------------|--------|-------|
| `position_lr_init` | 0.00016 | 0.00016 | 0.00016 |
| `position_lr_final` | 0.0000016 | 0.0000016 | 0.0000016 |
| `feature_lr` (DC) | 0.0025 | 0.0025 (lowfeature) | 0.0025 |
| `feature_lr` (rest) | 0.000125 | 0.00025 (highfeature/20) | 0.000125 |
| `opacity_lr` | 0.025 | 0.025 | 0.05 |
| `scaling_lr` | 0.005 | 0.005 | 0.005 |
| `rotation_lr` | 0.001 | 0.001 | 0.001 |
| `exposure_lr_init` | 0.01 | N/A | N/A |
| `exposure_lr_final` | 0.001 | N/A | N/A |

### 44.2 Densification Thresholds

| Parameter | gaussian-splatting | FastGS | AbsGS |
|-----------|-------------------|--------|-------|
| `percent_dense` | 0.01 | 0.001 | 0.001 |
| `densify_grad_threshold` | 0.0002 | 0.0002 | 0.0002 |
| `densify_grad_abs_threshold` | N/A | 0.0012 | 0.0004 |
| `loss_thresh` (metric) | N/A | 0.1 | N/A |
| `importance_threshold` | N/A | 5 | N/A |

### 44.3 Pruning Thresholds

| Parameter | gaussian-splatting | FastGS | AbsGS |
|-----------|-------------------|--------|-------|
| `min_opacity` (prune) | 0.005 | 0.005 | 0.005 |
| `opacity_reset_value` | 0.01 | 0.01 | 0.01 |
| `opacity_reduce_value` | N/A | 0.8 | 0.8 |
| `final_prune_opacity` | N/A | 0.1 | N/A |
| `final_prune_score` | N/A | 0.9 | N/A |
| `min_weight` | N/A | N/A | 0.7 |
| `prune_budget_ratio` | N/A | 0.5 | N/A |

---

## 45. Key Implementation Files Summary

### 45.1 Critical CUDA Files

| Implementation | File | Key Function |
|---------------|------|--------------|
| gaussian-splatting | `diff-gaussian-rasterization/forward.cu` | Original forward pass |
| gaussian-splatting | `diff-gaussian-rasterization/backward.cu` | Original backward pass |
| FastGS | `diff-gaussian-rasterization_fastgs/forward.cu` | Metric mode forward |
| FastGS | `diff-gaussian-rasterization_fastgs/adam.cu` | Sparse Adam |
| FastGS | `fused-ssim/ssim.cu` | Fused SSIM kernel |
| AbsGS | `diff-gaussian-rasterization-abs/backward.cu` | Homodirectional gradient |
| tinygs | `rasterizer/fastgs_ours/forward.cu` | Custom forward |
| tinygs | `rasterizer/fastgs_ours/kernels_backward.cuh` | Custom backward |
| tinygs | `loss/fused_ssim.cu` | Native fused SSIM |

### 45.2 Critical Python Files

| Implementation | File | Key Function |
|---------------|------|--------------|
| gaussian-splatting | `train.py` | Training loop with depth/exposure |
| gaussian-splatting | `scene/gaussian_model.py` | Gaussian parameter management |
| FastGS | `train.py` | Metric-based densification |
| FastGS | `scene/gaussian_model.py` | Dual optimizer, densify_and_prune_fastgs |
| FastGS | `utils/fast_utils.py` | compute_gaussian_score_fastgs |
| AbsGS | `train.py` | Weight tracking |
| AbsGS | `scene/gaussian_model.py` | Absolute gradient accumulation |

---

## 46. Known Limitations and Missing Features

### 46.1 tinygs vs References

| Feature | tinygs | Impact if Missing |
|---------|--------|-------------------|
| Depth regularization | ❌ | No geometry prior from depth maps |
| Exposure compensation | ❌ | Lower quality on varying exposure datasets |
| Anti-aliasing (EWA) | ❌ | Aliasing artifacts at different scales |
| GUI viewer | ❌ | No real-time visualization |
| Separate SH optimizer | ❌ | Slightly slower training |

### 46.2 Python References vs tinygs

| Feature | Python refs | Impact if Missing |
|---------|-------------|-------------------|
| FP16 training | ❌ (except tinygs) | Higher memory usage |
| Multiple rasterizers | ❌ | Less flexibility |
| Multiple strategies | ❌ | Harder to compare methods |
| NVTX profiling | ❌ | Harder to debug performance |
| Native fused SSIM | Partial (FastGS only) | More dependencies |

---

## 47. Recommended Configuration by Use Case

### 47.1 Fast Training (Speed Priority)

```
Strategy: FastGS
Rasterizer: fastgs_ours_fp16
Iterations: 30,000
Features:
  - Multi-view metric densification
  - Budget-based pruning
  - Reduced optimizer frequency after 15k
  - FP16 precision
```

### 47.2 High Quality (Quality Priority)

```
Strategy: AbsGS or default
Rasterizer: fastgs_ours
Iterations: 30,000
Features:
  - Absolute gradient for better splitting
  - Full optimizer steps throughout
  - FP32 precision
```

### 47.3 Memory Constrained

```
Strategy: FastGS with aggressive pruning
Rasterizer: fastgs_ours_fp16
Features:
  - FP16 training (2× memory reduction)
  - Lower importance threshold (more aggressive pruning)
  - Higher prune_budget_ratio
```

---

## 48. References and Sources

### 48.1 Original Papers

1. **3D Gaussian Splatting**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
2. **FastGS**: "FastGS: Speeding Up 3D Gaussian Splatting Training", 2024
3. **AbsGS**: "AbsGS: Recovering Fine Details for 3D Gaussian Splatting", 2024

### 48.2 Code Repositories

- gaussian-splatting: https://github.com/graphdeco-inria/gaussian-splatting
- FastGS: Referenced in ref_impl/FastGS/
- AbsGS: Referenced in ref_impl/AbsGS/

### 48.3 This Document

- Generated from analysis of ref_impl/ directory
- Line numbers accurate as of document creation
- tinygs implementation in tinygs/src/ and tinygs/include/

---

## 49. Detailed Forward Pass Blending Algorithm

### 49.1 Alpha Blending Equation

All implementations use the same alpha blending formula from the 3DGS paper (Eq. 3):

```cpp
// For each pixel, iterate Gaussians back-to-front (sorted by depth)
float T = 1.0f;        // Transmittance (starts at 1.0)
float3 color = {0, 0, 0};

for each Gaussian g in sorted_order:
    // Compute Gaussian influence at this pixel
    float d.x = pixel_x - g.mean2d.x;
    float d.y = pixel_y - g.mean2d.y;
    
    // Mahalanobis distance using inverse 2D covariance (conic)
    float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    
    if (power > -3.0f) {  // Culling threshold
        float G = exp(power);  // Gaussian value
        float alpha = min(0.99f, opacity * G);  // Apply opacity
        
        if (alpha >= 1.0f / 255.0f) {  // Skip very transparent
            float test_T = T * (1.0f - alpha);
            
            // Accumulate color
            color += T * alpha * g.color;
            T = test_T;
            
            // Early termination when nearly opaque
            if (T < 0.0001f) break;
        }
    }
}

// Add background
color += T * bg_color;
```

### 49.2 Key Blending Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `power_threshold` | -3.0 | Cull Gaussians with too little influence |
| `alpha_min` | 1/255 | Skip Gaussians below this alpha |
| `T_threshold` | 0.0001 | Early termination when transmittance is low |
| `alpha_cap` | 0.99 | Cap alpha to prevent numerical issues |

### 49.3 FastGS Metric Mode Integration

**During Forward Render** (forward.cu:401-407):
```cpp
// After accumulating color contribution
if (get_flag) {
    if (metric_map[pix_id] == 1) {
        // This pixel has high loss - count Gaussian contribution
        atomicAdd(&(metricCount[collected_id[j]]), 1);
    }
}
```

**Purpose**: Track which Gaussians contribute to high-error pixels across multiple views for densification guidance.

---

## 50. Exact SSIM Gaussian Kernel Weights

### 50.1 11-Tap Separable Gaussian (sigma ≈ 1.5)

From `ref_impl/FastGS/submodules/fused-ssim/ssim.cu:8-18`:

```cpp
#define G_00 0.001028380123898387f   // weight at offset -5
#define G_01 0.0075987582094967365f  // weight at offset -4
#define G_02 0.036000773310661316f   // weight at offset -3
#define G_03 0.10936068743467331f    // weight at offset -2
#define G_04 0.21300552785396576f    // weight at offset -1
#define G_05 0.26601171493530273f    // weight at offset 0 (center)
#define G_06 0.21300552785396576f    // weight at offset +1
#define G_07 0.10936068743467331f    // weight at offset +2
#define G_08 0.036000773310661316f   // weight at offset +3
#define G_09 0.0075987582094967365f  // weight at offset +4
#define G_10 0.001028380123898387f   // weight at offset +5
```

**Total sum**: 1.0 (normalized Gaussian)

### 50.2 SSIM Constants Derivation

```cpp
// From Wang et al. SSIM paper
constexpr float K1 = 0.01f;
constexpr float K2 = 0.03f;
constexpr float L = 1.0f;  // Dynamic range for [0,1] images

constexpr float C1 = (K1 * L) * (K1 * L);  // = 0.0001
constexpr float C2 = (K2 * L) * (K2 * L);  // = 0.0009
```

### 50.3 SSIM Formula

```
SSIM(x, y) = (2*μx*μy + C1)(2*σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))

Where:
- μx, μy: Local means (Gaussian-weighted average)
- σx², σy²: Local variances
- σxy: Local covariance
```

### 50.4 FastGS Fused SSIM Block Configuration

```cpp
#define BX 32  // Block width
#define BY 32  // Block height
#define SX (BX + 10)  // Shared memory width (block + 5-pixel halo each side)
#define SY (BY + 10)  // Shared memory height
```

---

## 51. Camera Model Implementation Details

### 51.1 Camera Parameters (All Implementations)

```python
# From gaussian-splatting/scene/cameras.py
class Camera:
    uid: int
    colmap_id: int
    R: np.ndarray       # 3x3 rotation matrix
    T: np.ndarray       # 3x1 translation vector
    FoVx: float         # Horizontal field of view
    FoVy: float         # Vertical field of view
    image_width: int
    image_height: int
    original_image: torch.Tensor  # [3, H, W]
    world_view_transform: torch.Tensor  # [4, 4]
    projection_matrix: torch.Tensor      # [4, 4]
    full_proj_transform: torch.Tensor    # [4, 4]
    camera_center: torch.Tensor          # [3]
```

### 51.2 Projection Matrix Construction

```python
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovX = tan((fovX / 2))
    tanHalfFovY = tan((fovY / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[3, 2] = -1.0
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    return P
```

### 51.3 Unique Camera Features

| Feature | gaussian-splatting | FastGS | AbsGS | tinygs |
|---------|-------------------|--------|-------|--------|
| Depth maps | ✅ | ❌ | ❌ | ❌ |
| Exposure params | ✅ | ❌ | ❌ | ❌ |
| Random background | Optional | ✅ | ✅ | ✅ |
| Alpha mask | ✅ | ❌ | ❌ | ❌ |

---

## 52. Detailed Backward Pass Gradient Equations

### 52.1 Gradient Chain for Gaussian Parameters

```
dL/dpixels (from loss)
    ↓
[Blend Backward]
    dL/dalpha = dL/dpixel_color * (T * color)  // For opacity
    dL/dcolor = dL/dpixel_color * T * alpha     // For SH
    dL/dmean2D = dL/dG * dG/ddelta              // For position
    dL/dconic = dL/dG * dG/dconic               // For scale/rotation
    ↓
[Preprocess Backward]
    dL/dmeans3D = J^T * dL/dmean2D              // Jacobian transpose
    dL/dscales = from dL/dconic3D
    dL/drotations = from dL/dconic3D
```

### 52.2 Gaussian Derivative Computation

From `backward.cu:541-542`:
```cpp
// Gaussian value: G = exp(-0.5 * (conic.x*dx² + 2*conic.y*dx*dy + conic.z*dy²))
// Derivatives:
const float dG_ddelx = -G * (conic.x * d.x + conic.y * d.y);
const float dG_ddely = -G * (conic.y * d.x + conic.z * d.y);
```

### 52.3 2D to 3D Gradient Propagation

```cpp
// Jacobian of projection: screen_pos = focal * world_pos / world_pos.z
// J = [f/z, 0, -f*x/z²]
//     [0, f/z, -f*y/z²]

// Chain rule: dL/dmeans3D = J^T * dL/dmean2D
dL_dmean3D.x = (focal_x / z) * dL_dmean2D.x;
dL_dmean3D.y = (focal_y / z) * dL_dmean2D.y;
dL_dmean3D.z = -(focal_x * x / z²) * dL_dmean2D.x - (focal_y * y / z²) * dL_dmean2D.y;
```

---

## 53. Spherical Harmonics Evaluation Details

### 53.1 SH Constants

From `auxiliary.h`:
```cpp
#define SH_C0 0.28209479177387814f
#define SH_C1[] = {0.4886025119029199f, -0.4886025119029199f, 0.4886025119029199f}
#define SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, ...}
#define SH_C3[] = {...}
```

### 53.2 SH Degree Progression Schedule

```python
# All implementations follow same schedule
if iteration % 1000 == 0:
    gaussians.oneupSHdegree()  # Increase active SH degree

# Maximum SH degree: 3 (48 total coefficients: 3 for DC, 45 for rest)
```

### 53.3 Direction Computation

```cpp
// From forward.cu:29-31
glm::vec3 pos = means[idx];
glm::vec3 dir = pos - campos;  // Direction from camera to Gaussian
dir = dir / glm::length(dir);  // Normalize
```

### 53.4 SH Color Evaluation (Degree 0-3)

```cpp
// Base color (DC term)
glm::vec3 result = SH_C0 * dc[idx];

if (deg > 0) {
    result += -SH_C1 * y * sh[0] + SH_C1 * z * sh[1] - SH_C1 * x * sh[2];
}
if (deg > 1) {
    result += SH_C2[0] * xy * sh[3] + SH_C2[1] * yz * sh[4] + ...;
}
if (deg > 2) {
    result += SH_C3[0] * y * (3*xx - yy) * sh[8] + ...;
}

result += 0.5f;  // Shift from [-0.5, 0.5] to [0, 1]
result = max(result, 0.0f);  // Clamp to positive
```

---

## 54. Tile-Based Rendering Pipeline Details

### 54.1 Tile Configuration

```cpp
// All implementations use same tile size
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
// Each tile is 16x16 pixels
```

### 54.2 Tile Sorting Pipeline

```
1. Preprocess (per Gaussian):
   - Project to 2D, compute conic, bounds
   - Determine tile overlap count
   
2. Create Instances:
   - For each tile a Gaussian touches, emit (tile_id, depth, gaussian_id)
   
3. Sort by Depth:
   - CUB radix sort on depth key
   
4. Sort by Tile:
   - CUB radix sort on tile_id key
   
5. Identify Ranges:
   - Find start/end indices for each tile in sorted list
   
6. Render (per tile, per block):
   - One CUDA block per tile
   - Each thread handles one pixel
   - Fetch Gaussians in sorted order, blend
```

### 54.3 FastGS Tile Bounding Box Multiplier

```cpp
// From forward.cu:246
uint32_t tiles_count = duplicateToTilesTouched(point_image, con_o, grid, mult, ...);

// mult = 0.5 (default in FastGS)
// Reduces tile bounding box for compact culling
// Lower mult = fewer tiles touched = faster but may miss coverage
```

---

## 55. Detailed Densification Criteria

### 55.1 Original 3DGS Criteria

```python
# Standard gradient threshold
grad_threshold = 0.0002

# Scale boundary for clone vs split
percent_dense = 0.01
scale_boundary = percent_dense * scene_extent

# Clone if: grad >= threshold AND max(scale) <= scale_boundary
# Split if: grad >= threshold AND max(scale) > scale_boundary
```

### 55.2 AbsGS Separate Thresholds

```python
# Separate gradient accumulators
xyz_gradient_accum      # Standard gradient (for clone)
xyz_gradient_accum_abs  # Absolute gradient (for split)

# Different thresholds
clone_threshold = 0.0002
split_threshold = 0.0004  # 2x higher

# Clone: norm(xyz_gradient_accum/denom) >= 0.0002 AND small scale
# Split: norm(xyz_gradient_accum_abs/denom) >= 0.0004 AND large scale
```

### 55.3 FastGS Multi-View Filter

```python
# Additional multi-view consistency filter
importance_score = compute_gaussian_score_fastgs(...)  # From 10 views
importance_threshold = 5

# Clone if: grad >= 0.0002 AND importance > 5 AND small scale
# Split if: absgrad >= 0.0012 AND importance > 5 AND large scale

# Budget-based pruning
prune_budget_ratio = 0.5  # Only remove 50% of prune candidates
```

---

## 56. Additional Algorithm Constants

### 56.1 Covariance Computation

```cpp
// EWA Splatting (Zwicker et al. 2002)
// 2D covariance from 3D covariance:
// Σ' = J * W * Σ * W^T * J^T
// where J is Jacobian of projection, W is world-to-view matrix

// Conic is inverse 2D covariance
// Used for Gaussian evaluation: G = exp(-0.5 * d^T * conic * d)
```

### 56.2 Scale Activation

```cpp
// Scales stored as log values, activated with exp
scale_activated = exp(scale_raw);  // Ensures positive values

// Default initialization
scale_raw = log(0.01 * point_cloud_average_spacing)
```

### 56.3 Rotation Normalization

```cpp
// Quaternions must be normalized for valid rotation
rotation_normalized = normalize(rotation_raw);

// During optimization, gradients may denormalize
// Re-normalize periodically or in backward pass
```

---

## 57. Numerical Stability Considerations

### 57.1 Gradient NaN Handling

```python
# From gaussian_model.py
grads = self.xyz_gradient_accum / self.denom
grads[grads.isnan()] = 0.0  # Replace NaN with 0
```

### 57.2 Alpha Clamping

```cpp
// Cap alpha to prevent numerical issues
float alpha = min(0.99f, opacity * G);

// Skip very transparent Gaussians
if (alpha < 1.0f / 255.0f) continue;
```

### 57.3 Depth Clipping

```cpp
// From forward.cu:87-92
const float limx = 1.3f * tan_fovx;
const float limy = 1.3f * tan_fovy;
t.x = min(limx, max(-limx, t.x / t.z)) * t.z;
t.y = min(limy, max(-limy, t.y / t.z)) * t.z;
// Clips points outside FoV to edge
```

### 57.4 Transmittance Threshold

```cpp
// Early termination when nearly opaque
if (T < 0.0001f) {
    done = true;
    break;
}
```

---

## 58. Implementation-Specific Optimizations

### 58.1 FastGS Speed Optimizations

1. **Metric-based Densification Control**:
   - Only densify Gaussians flagged by multi-view metric
   - Reduces unnecessary clone/split operations

2. **Reduced Optimizer Steps**:
   ```python
   # After 15k iterations, optimize less frequently
   if iteration <= 15000:
       optimizer.step()  # Every iteration
   elif iteration <= 20000:
       if iteration % 32 == 0: optimizer.step()
   else:
       if iteration % 64 == 0: optimizer.step()
   ```

3. **Separate SH Optimizer**:
   ```python
   # SH updated less frequently than other params
   if iteration % 16 == 0: shoptimizer.step()
   ```

### 58.2 AbsGS Floater Removal

1. **Weight Tracking**:
   ```python
   # Track alpha*T for each Gaussian
   gaussians.max_weight[visible] = max(gaussians.max_weight[visible], gs_w[visible])
   # Low max_weight = floater
   ```

2. **Opacity Reduction**:
   ```python
   # Cap opacity at 0.8 to help identify floaters
   opacities_new = inverse_sigmoid(min(get_opacity, 0.8))
   ```

### 58.3 tinygs Native Optimizations

1. **Tiled Image Layout** (8×8 tiles):
   ```cpp
   // Better memory coalescing for texture access
   uint idx = ((tile_y * width_in_tile + tile_x) << 6) | (in_tile_y << 3) | in_tile_x;
   ```

2. **Bucket-Based Blending**:
   ```cpp
   // Process 32 Gaussians per bucket
   // Store intermediate transmittance for backward pass
   // Reduces memory vs storing all intermediate values
   ```

3. **Dual Stream Execution**:
   - Stream 1: Forward pass
   - Stream 2: Loss + Backward preparation

---

## 59. Complete Hyperparameter Reference

### 59.1 Learning Rate Schedules

| Parameter | Init → Final | Schedule Type |
|-----------|--------------|---------------|
| `position_lr` | 0.00016 → 0.0000016 | Exponential decay |
| `feature_lr` (DC) | 0.0025 | Constant |
| `feature_lr` (rest) | 0.000125 | Constant (1/20 of DC) |
| `opacity_lr` | 0.05 | Constant |
| `scaling_lr` | 0.005 | Constant |
| `rotation_lr` | 0.001 | Constant |

### 59.2 Densification Schedule

| Phase | Iterations | Actions |
|-------|------------|---------|
| Warmup | 0-500 | No densification |
| Densify | 500-15000 | Clone/split/prune every 100 |
| Refine | 15000-30000 | No densification (original) or final prune (FastGS) |

### 59.3 Opacity Operations Schedule

| Operation | Interval | Value |
|-----------|----------|-------|
| Reset | Every 3000 | Cap at 0.01 |
| Reduce (AbsGS/FastGS) | Every 500 | Cap at 0.8 |
| Prune threshold | Continuous | < 0.005 |

---

## 60. Code Size and Complexity Metrics

### 60.1 CUDA Implementation Size

| Implementation | Forward (lines) | Backward (lines) | Total |
|---------------|-----------------|------------------|-------|
| gaussian-splatting | ~540 | ~660 | ~1200 |
| FastGS | ~543 | ~650 | ~1193 |
| AbsGS | ~540 | ~661 | ~1201 |
| tinygs fastgs_ours | ~600 | ~700 | ~1300 |

### 60.2 Python Code Size

| Implementation | gaussian_model.py (lines) | train.py (lines) |
|---------------|---------------------------|------------------|
| gaussian-splatting | 473 | 285 |
| FastGS | 540 | 286 |
| AbsGS | 483 | 248 |

---

## 61. Edge Cases and Special Handling

### 61.1 Empty Scene Handling

```python
# When no Gaussians visible in a view
if radii.sum() == 0:
    # Skip backward pass for that view
    continue
```

### 61.2 Single Gaussian Scene

```cpp
// Edge case: only one Gaussian
// Blending loop handles gracefully
// T will converge to background transmittance
```

### 61.3 Extreme Close-up Views

```cpp
// When Gaussian covers entire screen
// Tile overlap count may be very high
// Memory allocation should account for this
```

### 61.4 Transparent Background

```cpp
// When bg_color = [0, 0, 0] or [1, 1, 1]
// T contributes to final color
// background = T * bg_color
```

---

## 62. Debugging and Validation

### 62.1 Gradient Checking

```python
# Verify gradient computation
# Compare analytical gradient with numerical gradient
def check_gradient():
    epsilon = 1e-5
    numerical_grad = (loss(x + epsilon) - loss(x - epsilon)) / (2 * epsilon)
    assert abs(analytical_grad - numerical_grad) < 1e-4
```

### 62.2 Common Numerical Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| NaN gradients | Loss explodes | Check division by zero, clamp values |
| Opacity saturation | All opacities → 1 | Reset opacity periodically |
| Scale explosion | Gaussians too large | Add scale regularization |
| Rotation denormalization | Invalid quaternions | Re-normalize periodically |

---

## 63. Future Extension Points

### 63.1 Potential Additions to tinygs

1. **Depth Regularization**:
   - Requires depth map loading
   - Add depth loss term to training

2. **Exposure Compensation**:
   - Per-image exposure parameters
   - Separate optimizer for exposure

3. **Anti-aliasing (EWA)**:
   - Multi-scale rendering
   - Scale-dependent filtering

4. **GUI Viewer**:
   - Real-time network viewer
   - WebSocket-based communication

### 63.2 Algorithm Improvements

1. **Adaptive Learning Rates**:
   - Per-parameter learning rate schedules
   - Gradient norm-based adaptation

2. **Better Initialization**:
   - Structure-from-motion prior
   - Learning-based initialization

3. **Progressive Rendering**:
   - Start with low SH degree
   - Gradually increase complexity

---

## 64. Gaussian Initialization Methods

### 64.1 KNN-Based Initialization (All Python References)

All Python implementations use the same KNN-based initialization from `simple-knn`:

```cpp
// From simple_knn.cu
// Uses Morton code-based spatial hashing for fast KNN queries
#define BOX_SIZE 1024

// Morton code interleaving for 3D coordinates
__host__ __device__ uint32_t prepMorton(uint32_t x) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    return x;
}

// 3D Morton code (Z-order curve)
__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx) {
    uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * BOX_SIZE);
    uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * BOX_SIZE);
    uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * BOX_SIZE);
    return x | (y << 1) | (z << 2);
}
```

**Python Usage**:
```python
# From gaussian_model.py
from simple_knn._C import distCUDA2

def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    # Compute average distance to 3 nearest neighbors
    dist2 = distCUDA2(torch.from_numpy(pcd.points).float().cuda())
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    
    # Initialize positions from point cloud
    self._xyz = nn.Parameter(
        torch.tensor(pcd.points).float().cuda().requires_grad_(True))
    
    # Initialize scales from KNN distances (log space)
    self._scaling = nn.Parameter(scales.requires_grad_(True))
    
    # Initialize rotations as identity quaternions
    self._rotation = nn.Parameter(
        torch.zeros((fused_point_cloud.shape[0], 4), device="cuda"))
    self._rotation[:, 0] = 1  # w = 1, xyz = 0
    
    # Initialize opacities as low
    self._opacity = inverse_sigmoid(0.1 * torch.ones(...))
    
    # Initialize SH from point colors
    self._features_dc = nn.Parameter(
        RGB2SH(torch.tensor(pcd.colors).float().cuda()))
    self._features_rest = nn.Parameter(torch.zeros(...))
```

### 64.2 tinygs KNN Initialization

```cpp
// From tinygs/src/initialization/knn.cpp
// Uses nanoflann for KNN queries (CPU-based, more portable)

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, 
    PointCloudAdaptor, 3>;

std::vector<float> KnnInitialization::compute_mean_neighbor_distances(
    const std::vector<vec3>& points) const {
    
    PointCloudAdaptor cloud(points);
    KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    
    for (size_t i = 0; i < num_points; ++i) {
        // Find K nearest neighbors
        nanoflann::KNNResultSet<float> result_set(num_results);
        result_set.init(&ret_indices[0], &out_dists_sqr[0]);
        index.findNeighbors(result_set, &query_pt[0], ...);
        
        // Compute mean distance (skip self)
        float sum_dist = 0.0f;
        for (size_t j = 1; j < num_results; ++j) {
            sum_dist += out_dists_sqr[j];
        }
        result[i] = sqrt(sum_dist / valid_neighbors);
    }
}
```

### 64.3 Initialization Method Comparison

| Aspect | Python (simple-knn) | tinygs (nanoflann) |
|--------|--------------------|--------------------|
| **Algorithm** | Morton code + grid | KD-tree |
| **Execution** | CUDA GPU | CPU |
| **Dependencies** | Custom CUDA | nanoflann (header-only) |
| **Speed** | Fast for large clouds | Portable, no CUDA dep |
| **Neighbors** | 3 (fixed) | Configurable |

---

## 65. Python Binding Implementation Details

### 65.1 FastGS Rasterizer Binding

```cpp
// From rasterize_points.cu:52-76
std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& metric_map,      // FastGS specific
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& dc,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const float mult,                     // FastGS specific
    const bool prefiltered,
    const bool debug,
    const bool get_flag)                  // FastGS specific
```

### 65.2 AbsGS Rasterizer Binding

```cpp
// From rasterize_points.cu:35-55
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug)

// Key differences:
// 1. Returns gs_w (Gaussian weight for floater detection)
// 2. No metric_map/mult/get_flag parameters
// 3. dL_dmean2D is float4 (not float2) for abs gradient
```

### 65.3 Return Value Differences

| Return Value | gaussian-splatting | FastGS | AbsGS |
|--------------|-------------------|--------|-------|
| `out_color` | ✅ | ✅ | ✅ |
| `radii` | ✅ | ✅ | ✅ |
| `viewspace_points` | ✅ (float2) | ✅ (float2) | ✅ (float4!) |
| `visibility_filter` | ✅ | ✅ | ✅ |
| `accum_metric_counts` | ❌ | ✅ | ❌ |
| `gs_w` | ❌ | ❌ | ✅ |

### 65.4 Sparse Adam Binding (FastGS Only)

```cpp
// From rasterize_points.cu
void SparseAdamCUDA(
    torch::Tensor& param,
    const torch::Tensor& param_grad,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    const torch::Tensor& tiles_touched,  // Visibility mask
    const float lr,
    const float b1,
    const float b2,
    const float eps)
```

---

## 66. Viewspace Point Tensor Gradient Channels

### 66.1 Original 3DGS (float2)

```cpp
// Viewscape points: [N, 2] for 2D screen positions
// Gradient: [N, 2] for dL/dx, dL/dy
torch::Tensor dL_dmeans2D = torch::zeros({P, 2}, means3D.options());
```

### 66.2 AbsGS (float4)

```cpp
// Viewscape points: [N, 2] for 2D screen positions
// Gradient: [N, 4] for SIGNED + ABSOLUTE gradients!
torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());

// In backward.cu:544-550
// [0]: dL/dx (signed) - for clone decision
// [1]: dL/dy (signed) - for clone decision
// [2]: |dL/dx| (absolute) - for split decision
// [3]: |dL/dy| (absolute) - for split decision
```

### 66.3 Python Gradient Extraction

```python
# AbsGS gaussian_model.py:479-482
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    # Clone gradient: norm of first 2 components
    self.xyz_gradient_accum[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
    
    # Split gradient: norm of last 2 components
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(
        viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
```

---

## 67. Memory Buffer Management

### 67.1 Python Reference Memory Management

```python
# PyTorch automatic memory management
# Temporary buffers created per iteration
geomBuffer = torch.empty({0}, options.device(device))
binningBuffer = torch.empty({0}, options.device(device))
imgBuffer = torch.empty({0}, options.device(device))

// C++ side uses resizeFunctional for dynamic allocation
std::function<char*(size_t N)> geomFunc = resizeFunctional(geomBuffer);
```

### 67.2 tinygs Memory Pools

```cpp
// Persistent buffer pools for repeated allocations
struct PerPrimitiveBuffers {
    DoubleBuffer<uint> primitive_indices;  // Ping-pong for sorting
    DoubleBuffer<int> depth_keys;          // Ping-pong for sorting
    uint* n_touched_tiles;
    uint2* screen_bounds;
    float2* mean2d;
    float4* conic_opacity;
    float3* color;
    uint* offset;
    // CUB workspace managed separately
};

struct PerTileBuffers {
    uint2* instance_ranges;  // Start/end indices per tile
    uint* n_buckets;
    uint* bucket_offsets;
};

struct PerBucketBuffers {
    uint* tile_index;
    float4* color_transmittance;  // For backward pass
};
```

### 67.3 Memory Allocation Patterns

| Implementation | Strategy | Persistence |
|---------------|----------|-------------|
| Python refs | Per-iteration resize | Ephemeral |
| tinygs | Pre-allocated pools | Persistent |
| tinygs | Growth-only resize | No shrinking |

---

## 68. Additional FastGS Implementation Details

### 68.1 Multinomial Sampling for Budget Pruning

```python
# From gaussian_model.py:510-518
# Sample Gaussians to prune based on importance weights
scores = 1 - pruning_score  # Higher = more important to keep
to_remove = torch.sum(prune_mask)
remove_budget = int(0.5 * to_remove)  # Only remove 50%

if remove_budget:
    # Inverse: higher pruning_score = more likely to be sampled
    padded_importance = 1 / (1e-6 + scores.squeeze())
    
    # Weighted sampling without replacement
    sampled_indices = torch.multinomial(padded_importance, remove_budget, 
                                        replacement=False)
    selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool)
    selected_pts_mask[sampled_indices] = True
    
    final_prune = torch.logical_and(prune_mask, selected_pts_mask)
```

### 68.2 Final Prune Score Threshold

```python
# From gaussian_model.py:537-540
def final_prune_fastgs(self, min_opacity, pruning_score=None):
    prune_mask = (self.get_opacity < min_opacity).squeeze()
    scores_mask = pruning_score > 0.9  # High score = bad reconstruction
    final_prune = torch.logical_or(prune_mask, scores_mask)
    self.prune_points(final_prune)
```

---

## 69. Additional AbsGS Implementation Details

### 69.1 Initial Pruning Algorithm

```python
# From gaussian_model.py:360-372
def initial_prune(self):
    """Remove extremely large Gaussians at initialization."""
    # Criterion 1: Larger than mean scale
    pts_mask_1 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling)
    
    # Criterion 2: Top 0.1% or > 4x mean
    if len(self.get_scaling) < 5_000_000:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.quantile(
            self.get_scaling, 0.999)
    else:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.mean(
            self.get_scaling) * 4
    
    # Must match both criteria
    selected_pts_mask = torch.logical_and(pts_mask_1, pts_mask_2)
    self.prune_points(selected_pts_mask)
```

### 69.2 Weight-Based Floater Detection

```python
# From train.py:119-121
# Track per-Gaussian contribution (alpha * transmittance)
gs_w = render_pkg["gs_w"]
gaussians.max_weight[visibility_filter] = torch.max(
    gaussians.max_weight[visibility_filter],
    gs_w[visibility_filter])

# From train.py:138-142
# Prune Gaussians with low max contribution
if opt.use_prune_weight:
    prune_mask = (gaussians.max_weight < opt.min_weight).squeeze()  # 0.7
    gaussians.prune_points(prune_mask)
```

**Interpretation**:
- `gs_w` = α × T (alpha times transmittance) for each Gaussian
- High `max_weight`: Visible from many views with significant contribution → real geometry
- Low `max_weight`: Barely visible or low contribution → likely floater

---

## 70. Detailed Comparison of Gradient Accumulation

### 70.1 Standard Gradient (All Implementations)

```cpp
// In backward.cu during blend backward
// Standard signed gradient for position
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
```

### 70.2 Absolute Gradient (AbsGS/FastGS)

```cpp
// AbsGS backward.cu:548-550
// Additional absolute gradient for split decision
atomicAdd(&dL_dmean2D[global_id].z, fabs(dL_dG * dG_ddelx * ddelx_dx));
atomicAdd(&dL_dmean2D[global_id].w, fabs(dL_dG * dG_ddely * ddely_dy));
```

### 70.3 Why Absolute Gradient Helps Splitting

**Scenario**: A large Gaussian is under-reconstructed from multiple views

```
View 1: gradient points LEFT (x < 0)
View 2: gradient points RIGHT (x > 0)
View 3: gradient points LEFT (x < 0)

Standard gradient accumulation:
  (-0.5) + 0.5 + (-0.5) = -0.5  ← Low magnitude, Gaussian appears OK

Absolute gradient accumulation:
  |−0.5| + |0.5| + |−0.5| = 1.5  ← High magnitude, triggers split!

Result: Large under-reconstructed Gaussians correctly trigger splitting
```

---

## 71. Summary of All Key Differences

### 71.1 Architecture Level

| Aspect | Python Refs | tinygs |
|--------|-------------|--------|
| Language | Python + CUDA | C++20 + CUDA |
| Memory | PyTorch managed | Custom pools |
| Build | pip/setuptools | CMake |
| Dependencies | PyTorch, fused_ssim | Native implementations |

### 71.2 Algorithm Level

| Feature | gaussian-splatting | FastGS | AbsGS | tinygs |
|---------|-------------------|--------|-------|--------|
| Standard gradient | ✅ | ✅ | ✅ | ✅ |
| Absolute gradient | ❌ | ✅ | ✅ | ✅ |
| Multi-view metric | ❌ | ✅ | ❌ | ✅ |
| Weight tracking | ❌ | ❌ | ✅ | ❌ |
| Budget pruning | ❌ | ✅ | ❌ | ✅ |
| Depth regularization | ✅ | ❌ | ❌ | ❌ |
| Exposure compensation | ✅ | ❌ | ❌ | ❌ |

### 71.3 Implementation Level

| Feature | Python Refs | tinygs |
|---------|-------------|--------|
| FP16 training | ❌ | ✅ |
| Tiled image layout | ❌ | ✅ |
| NVTX profiling | ❌ | ✅ |
| Multiple rasterizers | ❌ | ✅ |
| Multiple strategies | ❌ | ✅ |
| Config-driven training | ❌ | ✅ |

---

## 72. MCMC Strategy Implementation (tinygs Exclusive)

### 72.1 Overview

The MCMC (Markov Chain Monte Carlo) strategy implements the approach from "3D Gaussian Splatting as Markov Chain Monte Carlo" paper. It uses a fundamentally different densification approach that avoids explicit pruning and instead uses relocation.

### 72.2 Key Algorithm Components

**1. Noise Injection** (mcmc.hpp:72):
```cpp
void add_noise(const RasterizeContext& ctx);
// Adds Gaussian noise to positions during optimization
// noise_lr = noise_lr_init * global_lr
// Default: noise_lr_init = 80.0 (equivalent to 5e5 * 1.6e-4)
```

**2. Sample Addition** (mcmc.hpp:73):
```cpp
void add_new_gs(const RasterizeContext& ctx);
// Adds new Gaussians proportional to current count
// grow_ratio = 1.05 (5% growth per densification step)
```

**3. Relocation Instead of Pruning** (mcmc.hpp:74):
```cpp
void relocate(const RasterizeContext& ctx);
// Instead of removing Gaussians, relocate them to high-gradient regions
// This maintains a relatively stable Gaussian count
```

### 72.3 MCMC-Specific Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `noise_lr_init` | 80.0 | Initial noise learning rate (scales with position LR) |
| `grow_ratio` | 1.05 | Growth ratio for adding new Gaussians |

### 72.4 Key Differences from Other Strategies

| Aspect | Standard Strategies | MCMC Strategy |
|--------|--------------------|--------------|
| Pruning | Remove low-opacity Gaussians | Relocate instead of prune |
| Gradient accumulation | Used for clone/split decisions | Not used |
| Noise | Not added | Added to positions each step |
| Gaussian count | Varies significantly | More stable |

---

## 73. Improved Strategy Implementation (tinygs Exclusive)

### 73.1 Overview

The Improved strategy combines elements from multiple approaches with additional enhancements:

1. Budget-based duplication control
2. Opacity reduction schedule
3. Position noise injection (from MCMC)
4. Distance-based splitting

### 73.2 Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `split_distance` | 0.3f | Distance threshold for splitting |
| `opacity_reduction` | 0.75f | Opacity reduction factor |
| `noise_lr_init` | 80.0f | Initial noise learning rate |

### 73.3 Algorithm Flow

```cpp
void step_impl(const RasterizeContext& ctx) override;
void duplicate(const RasterizeContext& ctx, int budget);  // Budget-controlled
void prune(const RasterizeContext& ctx);
```

---

## 74. Additional Loss Functions (tinygs Exclusive)

### 74.1 Available Loss Types

| Loss Type | Implementation | Description |
|-----------|---------------|-------------|
| `l1` | loss/l1.cu | L1 (MAE) loss |
| `l2` | loss/l2.cu | L2 (MSE) loss |
| `huber` | loss/huber.cu | Huber loss (smooth L1) |
| `fused_ssim` | loss/fused_ssim.cu | Native CUDA SSIM with backward |

### 74.2 Huber Loss Implementation

The Huber loss provides smooth L1 behavior:
```
L(x) = { 0.5 * x^2           if |x| <= delta
       { delta * (|x| - 0.5*delta)  otherwise
```

### 74.3 Loss Composition

```cpp
// From loss/loss.hpp
struct LossContext {
  Image pred;      // Predicted (rendered) image
  Image target;    // Ground-truth image
  Image loss;      // Per-pixel loss buffer (accumulated)
  Image grad;      // dL/d(pred) gradient buffer (accumulated)
  cudaStream_t stream;
};

// Multiple losses can be composed by accumulating into ctx.grad
virtual void evaluate(LossContext ctx, float scale) = 0;
```

---

## 75. tinygs Optimizer Architecture Details

### 75.1 Per-Parameter Learning Rates

```cpp
// From optim/optim.hpp
struct GaussianOptimizationParams {
  float means_lr = 1.6e-4f;      // Position LR
  float shs_lr = 2.5e-3f;        // Spherical harmonics LR
  float opacities_lr = 5.0e-2f;  // Opacity LR
  float scales_lr = 5.0e-3f;     // Scale LR
  float rotations_lr = 1.0e-3f;  // Rotation LR
  
  // Gradient clipping
  float max_grad_1 = 1.0f;       // L-inf clipping threshold
  bool skip_zero_grad = false;   // Skip updates for zero gradients
  
  // L1 regularization
  float opacities_l1 = 0.0f;
  float scales_l1 = 0.0f;
};
```

### 75.2 Optimizer Interface

```cpp
class OptimizerBase {
  // Standard step
  virtual void step(float scale, cudaStream_t stream) = 0;
  
  // Per-group control
  virtual void step(const GroupStepConfig& step_config, cudaStream_t stream);
  
  // State management after densification
  virtual void remove(char* kept_flag, int num_kept);
  virtual void duplicate(int* indices, int* new_indices, int num_duplicate);
  virtual void reset(int* indices, int num_reset) = 0;
  virtual void reorder(uint* indices) = 0;
  virtual void reset_opacity() = 0;
};
```

### 75.3 GroupStepConfig

```cpp
struct GroupStepConfig {
  bool update_means = true;
  bool update_shs = true;
  bool update_opacities = true;
  bool update_scales = true;
  bool update_rotations = true;
  
  float means_scale = 1.0f;
  float shs_scale = 1.0f;
  float opacities_scale = 1.0f;
  float scales_scale = 1.0f;
  float rotations_scale = 1.0f;
};
```

---

## 76. tinygs Rasterizer Architecture Details

### 76.1 RasterizeContext

```cpp
struct RasterizeContext {
  // Mode flags
  bool prepare_input_gradients = false;  // Compute camera gradients
  bool inference = false;                 // Skip backward storage
  bool metric_mode = false;               // Enable metric counting
  
  // Execution
  cudaStream_t stream = nullptr;
  float grad_scaler = 1.0f;               // FP16 training support
  
  // Input/Output
  GPUBatchInput fwd_input;     // Camera params + image dims
  GPUBatchOutput fwd_output;   // Rendered image
  GPUBatchInput grad_input;    // Camera gradients (output)
  GPUBatchOutput grad_output;  // Image gradients (input)
  
  // Densification info
  std::shared_ptr<GPUBuffer<DensificationInfo>> densification_info;
  
  // FastGS metric mode
  std::shared_ptr<GPUBuffer<int>> metric_map;    // Per-pixel flags
  std::shared_ptr<GPUBuffer<int>> metric_counts; // Per-Gaussian counts
};
```

### 76.2 Available Rasterizers

| Rasterizer | Description |
|------------|-------------|
| `default` | Basic 3DGS implementation |
| `fastgs` | FastGS-compatible interface |
| `fastgs_ours` | Custom FP32 implementation |
| `fastgs_ours_fp16` | Custom FP16 implementation |
| `3dgs_accel` | Accelerated original algorithm |

### 76.3 Rasterizer Interface

```cpp
class RasterizerBase {
  virtual void forward(const RasterizeContext& params) = 0;
  virtual void backward(RasterizeContext& params) = 0;
  virtual void forward_metric(const RasterizeContext& params);  // With metric mode
  virtual void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians);
};
```

---

## 77. DensificationInfo Structure

### 77.1 Fields

```cpp
struct DensificationInfo {
  float accum_grad_mean2d;       // Standard gradient (for clone)
  float accum_absgrad_mean2d;    // Absolute gradient (for split)
  float accum_counter;           // Visibility count
  float max_radii_screen;        // Max screen-space radius
  float metric_importance_score; // FastGS importance
  float metric_pruning_score;    // FastGS pruning priority
};
```

### 77.2 Usage in Strategies

- **Default**: Uses `accum_grad_mean2d` for both clone and split
- **AbsGS**: Uses `accum_grad_mean2d` for clone, `accum_absgrad_mean2d` for split
- **FastGS**: Same as AbsGS, plus `metric_importance_score` filtering
- **MCMC**: Does not use gradient accumulation
- **Improved**: Combined approach with additional heuristics

---

## 78. Strategy Parameters Summary

### 78.1 Default Strategy

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `duplicate_grad_threshold` | 0.0002 | Gradient threshold for densification |
| `duplicate_scale_threshold` | 0.01 | Scale boundary for clone/split |
| `refine_every` | 100 | Densification interval |
| `start_refine` | 500 | Start densification iteration |
| `end_refine` | 15000 | End densification iteration |
| `reset_every` | 3000 | Opacity reset interval |

### 78.2 AbsGS Strategy

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `m_absgrad_threshold` | 0.0004 | Split absolute gradient threshold |
| `m_percent_dense` | 0.001 | Scale boundary (clone/split) |

### 78.3 FastGS Strategy

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `m_absgrad_threshold` | 0.0012 | Split absolute gradient threshold |
| `m_percent_dense` | 0.001 | Scale boundary |
| `m_loss_thresh` | 0.1 | L1 threshold for metric map |
| `m_metric_num_cameras` | 10 | Cameras for multi-view scoring |
| `m_importance_threshold` | 5.0 | Minimum importance for densification |
| `m_prune_budget_ratio` | 0.5 | Fraction of candidates to prune |
| `m_final_prune_score_threshold` | 0.9 | Score for final prune |
| `m_final_prune_opacity_threshold` | 0.1 | Opacity for final prune |
| `m_final_prune_start` | 18000 | Start final prune |
| `m_final_prune_every` | 3000 | Final prune interval |

### 78.4 MCMC Strategy

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `noise_lr_init` | 80.0 | Initial noise LR (5e5 * 1.6e-4) |
| `grow_ratio` | 1.05 | Growth ratio per step |

### 78.5 Improved Strategy

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `m_split_distance` | 0.3 | Distance threshold for splitting |
| `m_opacity_reduction` | 0.75 | Opacity reduction factor |
| `m_noise_lr_init` | 80.0 | Initial noise LR |

---

## 79. Python vs tinygs: Detailed API Comparison

### 79.1 Training Loop (Python)

```python
# Gaussian-splatting
for iteration in range(iterations):
    gaussians.update_learning_rate(iteration)
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    loss = compute_loss(render_pkg["render"], gt_image)
    loss.backward()
    if iteration < densify_until_iter:
        gaussians.add_densification_stats(...)
        if iteration % densification_interval == 0:
            gaussians.densify_and_prune(...)
    gaussians.optimizer.step()
```

### 79.2 Training Loop (tinygs)

```cpp
// orchestrator.cu
for (int iteration = 0; iteration < config.iterations; ++iteration) {
  update_learning_rate(iteration);
  if (iteration % 1000 == 0) {
    gaussians->oneupSHdegree();
  }
  
  // Forward pass
  RasterizeContext ctx;
  ctx.fwd_input = camera_params;
  rasterizer->forward(ctx);
  
  // Loss computation
  LossContext loss_ctx;
  loss_ctx.pred = ctx.fwd_output.image;
  loss_ctx.target = gt_image;
  l1_loss->evaluate(loss_ctx, l1_weight);
  fused_ssim->evaluate(loss_ctx, ssim_weight);
  
  // Backward pass
  ctx.grad_output.image = loss_ctx.grad;
  rasterizer->backward(ctx);
  
  // Strategy step
  if (iteration < params.end_refine) {
    strategy->step(ctx);
  }
  
  // Optimizer step
  optimizer->step(grad_scale, stream);
}
```

---

## 80. Code Size Comparison

### 80.1 CUDA Implementation Size

| Component | gaussian-splatting | FastGS | AbsGS | tinygs |
|-----------|-------------------|--------|-------|--------|
| Forward kernel | ~300 lines | ~320 lines | ~300 lines | ~400 lines |
| Backward kernel | ~400 lines | ~420 lines | ~420 lines | ~500 lines |
| Preprocessing | ~100 lines | ~100 lines | ~100 lines | ~150 lines |
| Total | ~800 lines | ~840 lines | ~820 lines | ~1050 lines |

### 80.2 Python Code Size

| Component | gaussian-splatting | FastGS | AbsGS |
|-----------|-------------------|--------|-------|
| gaussian_model.py | 473 lines | 540 lines | 483 lines |
| train.py | 285 lines | 286 lines | 248 lines |
| arguments/__init__.py | 122 lines | 126 lines | 121 lines |

### 80.3 tinygs Code Size

| Component | Lines |
|-----------|-------|
| Strategy implementations | ~1500 lines total |
| Rasterizer implementations | ~2500 lines total |
| Optimizer implementations | ~800 lines |
| Loss implementations | ~600 lines |
| Core utilities | ~1000 lines |

---

## 81. Configuration Files Comparison

### 81.1 Python Configuration

```python
# Command-line arguments only
parser.add_argument('--densify_grad_threshold', type=float, default=0.0002)
parser.add_argument('--percent_dense', type=float, default=0.01)
```

### 81.2 tinygs Configuration (JSON)

```json
{
  "strategy": "fastgs",
  "rasterizer": "fastgs_ours",
  "iterations": 30000,
  "strategy_params": {
    "absgrad_threshold": 0.0012,
    "percent_dense": 0.001,
    "loss_thresh": 0.1,
    "metric_num_cameras": 10,
    "importance_threshold": 5.0,
    "prune_budget_ratio": 0.5
  },
  "optimizer_params": {
    "means_lr": 1.6e-4,
    "shs_lr": 2.5e-3,
    "opacities_lr": 5.0e-2,
    "scales_lr": 5.0e-3,
    "rotations_lr": 1.0e-3
  }
}
```

---

## 82. Scene-Specific Hyperparameter Tuning (FastGS Exclusive)

### 82.1 FastGS Scene-Specific Parameters

FastGS uses **different hyperparameters for different scenes**, discovered in training scripts (`train_base.sh`, `train_big.sh`):

| Scene | `grad_abs_thresh` | `highfeature_lr` | `loss_thresh` | `dense` | `mult` | `lowfeature_lr` |
|-------|-------------------|------------------|---------------|---------|--------|-----------------|
| **MipNeRF360** |
| bicycle | 0.0012 (base) / 0.0008 (big) | 0.005 | 0.1 | 0.001 | 0.5 | 0.0025 |
| flowers | 0.0015 (base) / 0.001 (big) | 0.005 | 0.1 | **0.005** | 0.5 | 0.0025 |
| garden | 0.0008 (base) / 0.0003 (big) | **0.02** | **0.06** | 0.001 | 0.5 | 0.0025 |
| stump | 0.0015 (base) / 0.001 (big) | 0.005 | 0.1 | **0.004** | 0.5 | 0.0025 |
| treehill | 0.002 (base) / 0.0018 (big) | 0.005 | 0.1 | **0.01** | 0.5 | 0.0025 |
| room | 0.0008 (base) / 0.0004 (big) | **0.02** | 0.1 | 0.001 | 0.5 | 0.0025 |
| counter | 0.0008 (base) / 0.0004 (big) | **0.02** | 0.1 | 0.001 | 0.5 | 0.0025 |
| kitchen | 0.0006 (base) / 0.0002 (big) | **0.02** | 0.1 | 0.001 | 0.5 | 0.0025 |
| bonsai | 0.0006 (base) / 0.0002 (big) | **0.02** | 0.1 | 0.001 | 0.5 | 0.0025 |
| **Tanks & Temples** |
| truck | 0.0009 (base) / 0.0004 (big) | **0.04** | 0.1 | 0.001 | **0.7** | 0.0025 |
| train | 0.0015 (base) / 0.0004 (big) | **0.042** | 0.1 | **0.01** | **0.7** | 0.0025 |
| **Deep Blending** |
| playroom | 0.0012 (base) / 0.0005 (big) | **0.0015** | 0.1 | **0.003** | **0.7** | 0.0025 |
| drjohnson | 0.0012 (base) / 0.0005 (big) | **0.0025** | 0.1 | **0.005** / **0.013** | **0.7** | **0.0005** (big) |

### 82.2 Key Observations

**Outdoor scenes** (MipNeRF360 garden, room, counter, kitchen, bonsai):
- Higher `highfeature_lr` (0.02) for better high-frequency detail capture
- Lower `grad_abs_thresh` (0.0002-0.0008) for more aggressive splitting

**Indoor/large scenes** (Tanks & Temples, Deep Blending):
- Higher `mult` (0.7) for larger tile bounding boxes (better coverage)
- Higher `highfeature_lr` (0.04) for Tanks & Temples
- Lower `highfeature_lr` (0.0015-0.0025) for Deep Blending
- Much higher `dense` values (0.003-0.013) for denser initialization

**Densification interval**:
- Base model: 500 iterations (faster training)
- Big model: 100 iterations (higher quality)

### 82.3 Parameter Tuning Guidelines

| Parameter | When to Increase | When to Decrease |
|-----------|------------------|------------------|
| `highfeature_lr` | Outdoor scenes with fine details | Indoor scenes, large-scale scenes |
| `grad_abs_thresh` | Reduce splitting, fewer Gaussians | More splitting, denser coverage |
| `dense` | Large-scale scenes needing denser coverage | Compact scenes with sufficient points |
| `mult` | Large scenes for better tile coverage | Small scenes for faster rendering |
| `loss_thresh` | Keep more Gaussians | More aggressive pruning |

---

## 83. Undocumented Features and Minor Changes (AbsGS)

### 83.1 Initial Pruning (Not in Paper)

**Purpose**: Remove extremely large Gaussians at initialization that are likely noise from COLMAP point cloud.

**Implementation** (ref_impl/AbsGS/scene/gaussian_model.py:360-372):
```python
def initial_prune(self):
    # Criterion 1: Larger than mean scale
    pts_mask_1 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling)
    
    # Criterion 2: Top 0.1% or > 4× mean
    if len(self.get_scaling) < 5_000_000:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.quantile(
            self.get_scaling, 0.999)
    else:
        pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.mean(
            self.get_scaling) * 4
    
    # Must match both criteria
    selected_pts_mask = torch.logical_and(pts_mask_1, pts_mask_2)
    self.prune_points(selected_pts_mask)
```

**Why This Matters**: COLMAP visualization filters points by reprojection error and track length, but the `.ply` file contains all points. This discrepancy leads to noisy initialization with extremely large Gaussians.

### 83.2 Weight-Based Pruning (Not in Paper)

**Purpose**: Prune Gaussians with low contribution during rendering.

**Implementation** (ref_impl/AbsGS/train.py:119-121, 138-142):
```python
# Track per-Gaussian contribution (alpha × transmittance)
gs_w = render_pkg["gs_w"]
gaussians.max_weight[visibility_filter] = torch.max(
    gaussians.max_weight[visibility_filter],
    gs_w[visibility_filter])

# Prune Gaussians with low max contribution
if opt.use_prune_weight:
    prune_mask = (gaussians.max_weight < opt.min_weight).squeeze()
    gaussians.prune_points(prune_mask)
```

**Default Parameters**:
- `use_prune_weight`: False (disabled by default)
- `min_weight`: 0.7
- `prune_until_iter`: 25000

### 83.3 Known Bug in Original 3DGS

**Issue**: The pruning strategy based on `max_radii2d` does not work in original 3DGS.

**Quote** (ref_impl/AbsGS/README.md:91):
> "In fact, the pruning strategy based on max_radii2d does not work for 3DGS, and we haven't fixed this bug."

**Impact**: This means the size-based pruning in original 3DGS is ineffective, potentially leading to Gaussians that are too large not being removed.

---

## 84. Additional Implementation Details

### 84.1 FastGS Config Typo

**File**: `ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/config.h:15`

```cpp
#define NUM_CHAFFELS 3 // Default 3, RGB
```

**Note**: Typo "NUM_CHAFFELS" instead of "NUM_CHANNELS". This appears in FastGS but not in other implementations (AbsGS correctly uses "NUM_CHANNELS").

### 84.2 SSIM Kernel Block Configuration

**FastGS Fused SSIM** (ref_impl/FastGS/submodules/fused-ssim/ssim.cu):
```cpp
#define BX 32  // Block width
#define BY 32  // Block height
#define SX (BX + 10)  // Shared memory width (32 + 10 = 42)
#define SY (BY + 10)  // Shared memory height (32 + 10 = 42)
```

**vs tinygs Fused SSIM**:
```cpp
// Uses 16×16 blocks instead of 32×32
// Tiled image layout (8×8 tiles)
// FP16 variant available
```

**Impact**: FastGS uses larger blocks (32×32 = 1024 threads) vs tinygs (16×16 = 256 threads), affecting:
- Shared memory requirements (42×42 vs 21×21)
- Register pressure
- Occupancy

### 84.3 Backward Pass Optimization Opportunity

**File**: `ref_impl/FastGS/submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/backward.cu:534`

```cpp
// TODO: perhaps store these things in shared memory?
if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
    T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
    int ii = i % 32;
    for (int ch = 0; ch < C; ++ch) 
        ar[ch] = -Shared_pixels[ch * 32 + ii] + Shared_sampled_ar[ch * 32 + ii];
```

**Note**: This TODO indicates a potential optimization where intermediate backward pass values could be stored in shared memory rather than global memory, potentially improving performance.

### 84.4 Opacity Clamping in Forward Pass

**All implementations** (ref_impl/*/cuda_rasterizer/forward.cu):
```cpp
float alpha = min(0.99f, con_o.w * exp(power));
```

**Purpose**:
- Cap alpha at 0.99 to prevent numerical issues
- Avoids alpha = 1.0 which would make transmittance = 0
- Ensures gradients can flow through all Gaussians

---

## 85. Performance Characteristics

### 85.1 FastGS Training Speed Claims

From FastGS README:
- **100 seconds** to train (vs 5-30 minutes for vanilla 3DGS)
- **3.32× faster** than DashGaussian on Mip-NeRF360
- **15.45× faster** than vanilla 3DGS on Deep Blending

### 85.2 Speed Optimization Techniques

**FastGS optimizations**:
1. **Reduced densification frequency** (500 vs 100 iterations) in base model
2. **Metric-based densification control** - only densify flagged Gaussians
3. **Reduced optimizer steps after 15k iterations**
4. **Separate SH optimizer** with lower frequency
5. **Higher `mult` (0.7)** for large scenes reduces tile overlap computation

### 85.3 Memory Characteristics

| Implementation | Typical VRAM | Notes |
|---------------|-------------|-------|
| vanilla 3DGS | High | All Gaussians updated every iteration |
| FastGS | Lower | Metric-based densification reduces Gaussian count |
| AbsGS | Similar to 3DGS | Weight tracking adds minimal overhead |
| tinygs FP32 | Similar | Tiled layout improves cache efficiency |
| tinygs FP16 | **50% lower** | Half-precision storage |

---

## 86. Dataset-Specific Behaviors

### 86.1 Resolution Handling

**All Python implementations**:
```python
# Automatic resizing if width > 1600 pixels
if resolution == -1:
    if image_width > 1600:
        # Automatically rescale to 1600 width
```

**tinygs**:
- Explicit resolution control via config
- No automatic resizing

### 86.2 Depth Map Support

**Original 3DGS only**:
```python
self._depths = ""  # Path to depth maps
```

- Loads depth maps from COLMAP
- Uses depth regularization loss
- FastGS and AbsGS remove this feature

### 86.3 Exposure Compensation

**Original 3DGS only**:
```python
self.train_test_exp = False  # Per-image exposure optimization
```

- Learns per-image 3×4 affine color transform
- Compensates for varying exposure across training images
- FastGS and AbsGS remove this feature

---

## 87. Implementation-Specific Constants

### 87.1 Tile Configuration

| Implementation | Tile Size | Block Size |
|---------------|-----------|------------|
| All Python refs | 16×16 | 256 threads |
| tinygs | 16×16 | 256 threads |

### 87.2 SSIM Constants

**All implementations use same constants**:
```cpp
constexpr float C1 = 0.0001f;  // (K1 * L)² = (0.01 * 1)²
constexpr float C2 = 0.0009f;  // (K2 * L)² = (0.03 * 1)²
```

### 87.3 Gaussian Kernel Weights (SSIM)

**11-tap separable Gaussian** (sigma ≈ 1.5):
```cpp
// Symmetric weights, center at G_05
G_00 = 0.001028...  // offset -5
G_01 = 0.007598...  // offset -4
G_02 = 0.036000...  // offset -3
G_03 = 0.109360...  // offset -2
G_04 = 0.213005...  // offset -1
G_05 = 0.266011...  // center
```

**Total sum**: 1.0 (normalized)

---

## 88. Code Quality and Maintenance

### 88.1 Code Annotations

**TODOs found**:
- FastGS backward.cu:534: "perhaps store these things in shared memory?"

**No FIXMEs or HACKs found** in CUDA implementation files.

### 88.2 Error Handling

**Python references**:
```python
try:
    with open(cfgfilepath) as cfg_file:
        cfgfile_string = cfg_file.read()
except TypeError:
    print("Config file not found")
    pass
```

**tinygs**:
- Uses `log_*` macros for all errors
- `CHECK_THROW()` for validation
- `CUDA_CHECK_THROW()` for CUDA errors

### 88.3 Debug Support

**All implementations** support:
- `--debug` flag for detailed error reporting
- Dump files when rasterizer fails
- `--debug_from` to enable debugging after specific iteration

---

## 89. Testing and Validation

### 89.1 Evaluation Protocols

**All Python implementations**:
```python
# Test at specific iterations
--test_iterations 7000 30000

# Compute metrics
python metrics.py -m <model_path>
```

**Metrics computed**:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

### 89.2 Full Evaluation Script

**Python references** provide `full_eval.py`:
```bash
python full_eval.py \
    -m360 <mipnerf360_folder> \
    -tat <tanks_and_temples_folder> \
    -db <deep_blending_folder>
```

**tinygs**:
- No built-in full evaluation script
- Relies on external metrics computation

---

## 90. Differences in Training Scripts

### 90.1 FastGS Dual Training Modes

**Base model** (train_base.sh):
```bash
--densification_interval 500  # Faster training
```

**Big model** (train_big.sh):
```bash
--densification_interval 100  # Higher quality
```

### 90.2 Scene-Specific Tuning

**FastGS scripts show extensive per-scene tuning**:
- Different `grad_abs_thresh` for each scene
- Different `highfeature_lr` for indoor vs outdoor
- Different `mult` for large-scale scenes

**AbsGS and vanilla 3DGS**:
- Single set of hyperparameters
- No scene-specific tuning scripts

---

## 91. File Organization Differences

### 91.1 Submodule Structure

**All Python implementations**:
```
submodules/
├── diff-gaussian-rasterization-*/  # CUDA rasterizer
│   ├── cuda_rasterizer/
│   │   ├── forward.cu
│   │   ├── backward.cu
│   │   ├── auxiliary.h
│   │   └── config.h
│   └── rasterize_points.cu  # Python binding
├── simple-knn/  # KNN for initialization
│   └── simple_knn.cu
└── fused-ssim/  # FastGS only
    └── ssim.cu
```

**tinygs**:
```
tinygs/src/
├── rasterizer/
│   ├── fastgs_ours/
│   │   ├── forward.cu
│   │   └── kernels_backward.cuh
│   └── 3dgs_accel/
├── strategy/
│   ├── default.cu
│   ├── fastgs.cu
│   └── absgs.cu
├── loss/
│   └── fused_ssim.cu
└── initialization/
    └── knn.cpp
```

---

## 92. Build and Dependency Management

### 92.1 Python Dependencies

**All Python implementations**:
```yaml
# environment.yml
name: fastgs  # or absgs, 3dgs
dependencies:
  - python=3.8
  - pytorch
  - cuda
  - plyfile
  - tqdm
```

**Additional for FastGS**:
- `fused-ssim` (separate CUDA extension)

### 92.2 tinygs Dependencies

**CMake-based**:
```cmake
# CPM.cmake for dependency management
CPMAddPackage("gh:catchorg/Catch2@3.4.0")
CPMAddPackage("gh:g-truc/glm#0.9.9.8")
# ... others
```

**External libraries** (vendored):
- OpenCV
- spdlog
- nlohmann_json
- glm
- NVTX3
- cxxopts
- Thrust (CUDA)

---

## 93. Known Limitations by Implementation

### 93.1 Original 3DGS Limitations

1. **max_radii2d pruning bug**: Does not work correctly
2. **No absolute gradient**: Uses same gradient for clone and split
3. **High memory usage**: All Gaussians updated every iteration

### 93.2 FastGS Limitations

1. **Scene-specific tuning required**: Not one-size-fits-all
2. **Removed depth regularization**: No geometry prior
3. **Removed exposure compensation**: No per-image exposure handling
4. **Config typo**: "NUM_CHAFFELS" instead of "NUM_CHANNELS"

### 93.3 AbsGS Limitations

1. **Removed depth regularization**: No geometry prior
2. **Removed exposure compensation**: No per-image exposure handling
3. **Weight-based pruning not in paper**: Undocumented feature

### 93.4 tinygs Limitations

1. **No GUI viewer**: No real-time visualization
2. **No depth regularization**: Not implemented
3. **No exposure compensation**: Not implemented
4. **No anti-aliasing**: No EWA filter

---

## 94. Summary of All Discovered Details

### 94.1 Critical Findings

1. **FastGS uses extensive scene-specific hyperparameter tuning** - not documented in paper
2. **AbsGS has undocumented features** (initial pruning, weight-based pruning) not in paper
3. **Known bug in 3DGS**: max_radii2d-based pruning doesn't work
4. **FastGS typo**: NUM_CHAFFELS in config.h
5. **Optimization opportunity**: FastGS backward pass TODO for shared memory

### 94.2 Implementation Quality

| Aspect | Original 3DGS | FastGS | AbsGS | tinygs |
|--------|--------------|--------|-------|--------|
| **Code comments** | Minimal | Minimal | Minimal | High verbosity |
| **TODOs** | None | 1 | None | None |
| **Known bugs** | 1 (max_radii2d) | 0 | 0 | 0 |
| **Documentation** | Good | Good | Good | Excellent (AGENTS.md) |
| **Test coverage** | Basic | Basic | Basic | Comprehensive |

### 94.3 Parameter Tuning Effort

| Implementation | Tuning Effort | Scene-Specific? |
|---------------|---------------|-----------------|
| Original 3DGS | Low | No |
| FastGS | **High** | **Yes** (13+ scenes) |
| AbsGS | Low | No |
| tinygs | Medium | No (config-driven) |

---

## 95. Recommendations for tinygs Users

### 95.1 Choosing a Strategy

**For fastest training**:
```
Strategy: FastGS
Rasterizer: fastgs_ours_fp16
Config: Use scene-specific params from train_big.sh
```

**For highest quality**:
```
Strategy: AbsGS or default
Rasterizer: fastgs_ours
Iterations: 30,000
```

**For balanced speed/quality**:
```
Strategy: FastGS
Rasterizer: fastgs_ours
Densification interval: 100-200
```

### 95.2 Hyperparameter Tuning Guidelines

**Based on FastGS scene-specific tuning**:

| Scene Type | `highfeature_lr` | `grad_abs_thresh` | `dense` | `mult` |
|------------|------------------|-------------------|---------|--------|
| Outdoor (nature) | 0.02 | 0.0002-0.0008 | 0.001 | 0.5 |
| Outdoor (urban) | 0.04 | 0.0004-0.0009 | 0.001 | 0.7 |
| Indoor (room) | 0.02 | 0.0004-0.0008 | 0.001 | 0.5 |
| Indoor (large) | 0.0015-0.0025 | 0.0005-0.0012 | 0.003-0.013 | 0.7 |

### 95.3 Memory Optimization

**Reduce memory usage**:
1. Use `fastgs_ours_fp16` rasterizer
2. Lower `importance_threshold` for more aggressive pruning
3. Increase `prune_budget_ratio` to remove more Gaussians
4. Use `data_device: cpu` for large datasets (Python only)

---

## 96. tinygs Exclusive Features (Not in Python References)

### 96.1 Gaussian Reordering via Morton Code

**Purpose**: Improve memory locality during rendering by sorting Gaussians spatially.

**Implementation** (orchestrator.cu:54-128):
```cpp
// Morton code computation for 3D spatial ordering
thrust::transform(positions, positions + n, enc_in.begin(), 
  [min_pos, inv_dx, inv_dy, inv_dz] __device__ (const vec3& p) -> uint {
    // Normalize position to [0,1] within bounding box
    const float nx = fminf(fmaxf((p.x - min_pos.x) * inv_dx, 0.0f), 1.0f);
    const float ny = fminf(fmaxf((p.y - min_pos.y) * inv_dy, 0.0f), 1.0f);
    const float nz = fminf(fmaxf((p.z - min_pos.z) * inv_dz, 0.0f), 1.0f);
    
    // Map to 10-bit integer grid and compute Morton code
    const uint32_t xi = static_cast<uint32_t>(nx * 1023.0f);
    const uint32_t yi = static_cast<uint32_t>(ny * 1023.0f);
    const uint32_t zi = static_cast<uint32_t>(nz * 1023.0f);
    return morton3D(xi, yi, zi);
  });

// Radix sort by Morton code
cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
  enc_in.data(), enc_out.data(), idx_in.data(), idx_out.data(), n, 0, 30, stream);
```

**Benefits**:
- Better cache locality during tile-based rendering
- Improved memory coalescing for Gaussian attribute access
- Periodic reordering maintains spatial locality as Gaussians move during training

**Configuration** (configs/garden.json):
```json
{
  "trainer": {
    "reorder_gaussians_interval": 1000
  }
}
```

### 96.2 Async Dataloader

**Purpose**: Overlap data loading with GPU computation for improved throughput.

**Implementation**:
```cpp
struct DataloaderConfig {
  std::string type = "default";  // "default" or "async"
  std::string data_type = "float32";  // "float32" or "float16"
};
```

**Benefits**:
- Prefetches next training image while GPU processes current iteration
- Reduces CPU-GPU synchronization overhead
- Particularly beneficial for high-resolution images

**Configuration**:
```json
{
  "dataloader": {
    "type": "async",
    "data_type": "float16"
  }
}
```

### 96.3 Camera Pose Optimization

**Purpose**: Refine camera poses jointly with Gaussians during training.

**Implementation**:
```cpp
struct PoseOptConfig {
  std::string type = "adamw";
  float lr = 0.0001f;
  float momentum = 0.95f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;
  float weight_decay = 0.1f;
};
```

**Configuration**:
```json
{
  "pose_opt": {
    "type": "adamw",
    "lr": 0.0001,
    "weight_decay": 0.1
  },
  "trainer": {
    "start_pose_opt": 50000
  }
}
```

**Notes**:
- Pose optimization starts after Gaussian structure stabilizes
- Uses AdamW optimizer with weight decay for regularization
- Not available in Python reference implementations

### 96.4 Gradient Accumulation Configuration

**Purpose**: Control optimization frequency per parameter group.

**Configuration**:
```json
{
  "trainer": {
    "means_accumulate_grad_steps": 1,
    "shs_accumulate_grad_steps": 16,
    "opacities_accumulate_grad_steps": 1,
    "scales_accumulate_grad_steps": 1,
    "rotations_accumulate_grad_steps": 1
  }
}
```

**Comparison with FastGS**:
| Parameter | FastGS (hardcoded) | tinygs (configurable) |
|-----------|-------------------|----------------------|
| SH update frequency | Every 16 iterations (0-15k) | Configurable via `shs_accumulate_grad_steps` |
| Main optimizer frequency | Every iteration (0-15k) | Configurable per parameter |

### 96.5 Additional Training Features

#### Early Stopping
```json
{
  "trainer": {
    "enable_early_stopping": false,
    "early_stopping_threshold": 1e-6,
    "early_stopping_patience": 1000
  }
}
```

#### Time-Based Training Limit
```json
{
  "trainer": {
    "max_seconds": 240
  }
}
```
- Stops training after specified time (useful for benchmarking)
- Not available in Python references

#### FastGS-Style Learning Rate Schedule
```json
{
  "lr_schedulers": {
    "means": {
      "type": "exponential",
      "use_fastgs_schedule": true
    }
  }
}
```

---

## 97. Configuration File Format Comparison

### 97.1 Python Reference Configuration

All Python implementations use command-line arguments:
```bash
python train.py -s <data_path> --densify_grad_threshold 0.0002 --percent_dense 0.01
```

### 97.2 tinygs JSON Configuration

```json
{
  "dataset": {
    "root_path": "/path/to/data",
    "type": "image"
  },
  "rasterizer": {
    "type": "fastgs_ours"
  },
  "strategy": {
    "type": "fastgs",
    "duplicate_grad_threshold": 0.0008,
    "absgrad": true
  },
  "optimizer": {
    "type": "adam",
    "means_lr": 0.00016
  },
  "losses": [
    {"type": "l1", "weight": 0.8},
    {"type": "fused_ssim", "weight": 0.2}
  ]
}
```

### 97.3 Configuration Advantages

| Aspect | Python (CLI) | tinygs (JSON) |
|--------|-------------|---------------|
| Reproducibility | Requires script logging | Self-documenting config |
| Parameter validation | Runtime errors | Schema validation possible |
| Scene-specific configs | Separate shell scripts | One JSON file per scene |
| Version control | Difficult to track | Easy to diff and version |
| Extensibility | Requires code changes | New fields easily added |

---

## 98. Implementation Differences Summary

### 98.1 What tinygs Has That Python References Don't

| Feature | tinygs | Notes |
|---------|--------|-------|
| FP16 training | ✅ | 50% memory reduction |
| Gaussian reordering | ✅ | Morton code spatial ordering |
| Async dataloader | ✅ | CPU-GPU overlap |
| Pose optimization | ✅ | Joint camera refinement |
| Configurable gradient accumulation | ✅ | Per-parameter control |
| Time-based training limit | ✅ | `max_seconds` parameter |
| Early stopping | ✅ | Configurable patience |
| Multiple rasterizer backends | ✅ | 4 different implementations |
| Multiple strategy implementations | ✅ | 5 different strategies |
| NVTX profiling | ✅ | Built-in profiling ranges |

### 98.2 What Python References Have That tinygs Doesn't

| Feature | Python refs | Notes |
|---------|-------------|-------|
| Depth regularization | ✅ (original only) | Geometry prior from depth maps |
| Exposure compensation | ✅ (original only) | Per-image exposure optimization |
| Anti-aliasing (EWA) | ✅ (original only) | Multi-scale rendering |
| GUI viewer | ✅ (all) | Real-time network viewer |
| Separate SH optimizer | ✅ (FastGS) | Dedicated SH learning rate schedule |

---

## 99. Practical Training Recommendations

### 99.1 Fast Training (Speed Priority)

```json
{
  "rasterizer": {"type": "fastgs_ours_fp16"},
  "strategy": {"type": "fastgs"},
  "trainer": {
    "max_steps": 30000,
    "train_data_type": "float16",
    "shs_accumulate_grad_steps": 16
  }
}
```

### 99.2 High Quality (Quality Priority)

```json
{
  "rasterizer": {"type": "fastgs_ours"},
  "strategy": {"type": "absgs"},
  "trainer": {
    "max_steps": 30000,
    "shs_accumulate_grad_steps": 1
  }
}
```

### 99.3 Memory-Constrained Training

```json
{
  "rasterizer": {"type": "fastgs_ours_fp16"},
  "strategy": {
    "type": "fastgs",
    "max_num_gaussians": 500000,
    "pruning_opacity_threshold": 0.02
  },
  "trainer": {
    "train_data_type": "float16",
    "eval_data_type": "float16"
  }
}
```

---

## 100. Final Summary

This document represents a comprehensive analysis of the **tinygs** implementation compared to three reference implementations (**gaussian-splatting**, **FastGS**, **AbsGS**), covering:

- **Architecture differences**: C++/CUDA vs Python, native vs PyTorch bindings
- **Algorithm details**: Densification strategies, gradient computation, rendering pipeline
- **Implementation specifics**: Memory layout, kernel optimizations, SSIM computation
- **Hyperparameter tuning**: Default values, scene-specific parameters, thresholds
- **Undocumented features**: Initial pruning, weight tracking, known bugs
- **tinygs exclusives**: FP16 training, Gaussian reordering, async dataloader, pose optimization

Key takeaways:
1. **tinygs** provides significant performance advantages through native C++/CUDA implementation
2. **FastGS** achieves 100-second training through aggressive optimization and scene-specific tuning
3. **AbsGS** improves quality through homodirectional gradient for better split decisions
4. Each implementation has trade-offs between speed, quality, and features
