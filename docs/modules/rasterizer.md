# Rasterizer Module

The rasterizer module handles forward and backward rendering of 3D Gaussians to 2D images.

## Files

| File | Description |
|------|-------------|
| `rasterizer.hpp` | Base class and context structures |
| `default.hpp` | Reference 3DGS implementation |
| `fastgs.hpp` | FastGS approximation |
| `gsplat.hpp` | GSplat implementation |

## Implementations

| Type | Description | Speed | Quality |
|------|-------------|-------|---------|
| `default` | Reference 3DGS | Baseline | Best |
| `fastgs` | FastGS approximation | Faster | Good |
| `gsplat` | GSplat implementation | Fast | Good |

---

## RasterizeContext

Container for all data needed during rasterization:

```cpp
struct RasterizeContext {
    bool prepare_input_gradients = false;  // Compute camera gradients
    bool inference = false;                 // Skip backward storage
    cudaStream_t stream = nullptr;          // CUDA stream
    float grad_scaler = 1.0f;               // Gradient scaling (128.0 for fp16)
    
    GPUBatchInput fwd_input;     // Camera parameters
    GPUBatchOutput fwd_output;   // Rendered image
    GPUBatchInput grad_input;    // Input gradients
    GPUBatchOutput grad_output;  // Output gradients
    std::shared_ptr<GPUGaussian3d> gaussians_grad;  // Gaussian gradients
    
    mutable std::shared_ptr<GPUBuffer<DensificationInfo>> densification_info;
};
```

---

## RasterizerBase Interface

```cpp
class RasterizerBase {
public:
    virtual ~RasterizerBase() = default;
    
    /// @brief Render Gaussians to image
    virtual void forward(const RasterizeContext& params) = 0;
    
    /// @brief Backpropagate gradients to Gaussians
    virtual void backward(RasterizeContext& params) = 0;
    
    /// @brief Update Gaussian reference
    virtual void set_gaussians(std::shared_ptr<GPUGaussian3d> gaussians);
    
    /// @brief Configuration management
    virtual json get_params() const = 0;
    virtual void set_params(const json& j) = 0;
};
```

---

## Forward Pass

Renders 3D Gaussians to a 2D image using alpha-blending:

1. **Projection**: Project 3D Gaussians to 2D screen space
2. **Sorting**: Sort Gaussians by depth (back-to-front)
3. **Tile Assignment**: Assign Gaussians to screen tiles
4. **Rendering**: Alpha-blend Gaussians per pixel

```cpp
// Forward rendering
RasterizeContext ctx;
ctx.fwd_input = batch_input;  // Camera intrinsics/extrinsics
ctx.fwd_output.image = render_buffer;  // Output image
ctx.gaussians_grad = gradients;  // For backward storage
rasterizer->forward(ctx);

// Result is in ctx.fwd_output.image
```

---

## Backward Pass

Computes gradients with respect to Gaussian parameters:

1. **Gradient Accumulation**: Accumulate per-pixel gradients
2. **Backpropagation**: Propagate gradients through blending
3. **Parameter Gradients**: Compute gradients for:
   - `means`: 3D positions
   - `opacities`: Alpha values
   - `scales`: Size parameters
   - `rotations`: Orientation quaternions
   - `sh_coefficient_0`: DC color
   - `sh_coefficients_rest`: Higher-order SH coefficients

```cpp
// Backward pass
ctx.grad_output.image = image_gradients;  // From loss
rasterizer->backward(ctx);

// Gradients are in ctx.gaussians_grad
```

---

## GPUBatchInput

Camera data for rendering:

```cpp
struct GPUBatchInput {
    uint32_t width, height;  // Image dimensions
    float near, far;         // Clipping planes
    mat3x3 K;               // Intrinsic matrix
    mat4x4 w2c;             // World-to-camera matrix
    uuid_t timestamp;       // Frame identifier
};
```

---

## GPUBatchOutput

Rendering output:

```cpp
struct GPUBatchOutput {
    Image image;  // Rendered RGB image (CHW format)
};
```

---

## DensificationInfo

Per-pixel information used by strategies:

```cpp
struct DensificationInfo {
    float position_gradient_accum;  // Accumulated position gradient
    float denom;                    // Normalization factor
};
```

Used by strategies to determine where to split or clone Gaussians.

---

## Creating a Rasterizer

```cpp
// Via factory
auto rasterizer = create_rasterizer("fastgs");

// Configure
json params = {{"type", "fastgs"}};
rasterizer->set_params(params);

// Set Gaussians
rasterizer->set_gaussians(gaussians);
```

---

## Implementation Details

### Default Rasterizer

Reference implementation following the original 3DGS paper:

- Full sorting of Gaussians per tile
- Accurate alpha-blending
- Complete gradient computation

### FastGS Rasterizer

Optimized implementation with approximations:

- Faster tile-based culling
- Approximated blending
- FP16 gradient scaling support

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f16_grad_scaler` | float | 1.0 | Gradient scaler for fp16 |
| `enable_pose_opt` | bool | false | Enable camera pose gradients |

---

## Memory Management

Rasterizers use a memory arena pattern for intermediate buffers:

```cpp
class RasterizerBase {
protected:
    std::shared_ptr<GPUMemoryArena> m_memory_arena;
};
```

Buffers are reused across forward/backward passes to minimize allocations.

---

## Indexing Bugs and Limitations

### Known Issues

1. **FP16 key truncation (fastgs_ours_fp16)**
   - **Location:** `tinygs/src/rasterizer/fastgs_ours_fp16/kernels_forward.cuh:538`
   - **Problem:** `tile_key` is written as `ushort` into a `uint` key buffer, while the cooperative path uses full `uint` at line 586. If tile index exceeds 65535, keys overflow/corrupt ordering.
   - **Impact:** Incorrect tile ordering for grids with >65535 tiles (e.g., >1044×1044 at 16×16 tiles).

2. **Key-width limit (fastgs_ours and fastgs_ours_fp16)**
   - **Location:** `tinygs/src/rasterizer/fastgs_ours/kernels_forward.cuh:496` and `tinygs/src/rasterizer/fastgs_ours_fp16/kernels_forward.cuh:540`
   - **Problem:** Keys are stored as `ushort`, and radix sort uses 16-bit width at `tinygs/src/rasterizer/fastgs_ours/forward.cu:219`.
   - **Impact:** Hard-limits safe tile index range to 65535. Larger grids will fail or produce incorrect results.

3. **Screen-bounds storage cap**
   - **Location:** `tinygs/src/rasterizer/fastgs_ours/kernels_forward.cuh:406` (FP32) and `tinygs/src/rasterizer/fastgs_ours_fp16/kernels_forward.cuh:437` (FP16)
   - **Problem:** Tile bounds are cast to `ushort` in `primitive_screen_bounds`. Large grids can overflow.
   - **Impact:** Truncated bounds for grids with >65535 tiles. Debug asserts catch this, but release builds silently truncate.

### Metric Mode Indexing

- **Correct behavior:** Rendered/GT images use 8×8 tiled storage (`get_linear_index_tiled`), while `metric_map` and `metric_counts` remain flat row-major (`y * width + x`).
- **Implementation:** `tinygs/src/strategy/fastgs.cu:66-93` correctly uses tiled indexing for L1 computation and flat indexing for metric map access.

---

## Usage Example

```cpp
// Setup
auto rasterizer = create_rasterizer("fastgs");
rasterizer->set_gaussians(gaussians);

// Forward
RasterizeContext ctx;
ctx.stream = cuda_stream;
ctx.fwd_input = camera_data;
ctx.fwd_output.image = render_target;
rasterizer->forward(ctx);

// Compute loss and get image gradients
// ...

// Backward
ctx.grad_output.image = image_gradients;
rasterizer->backward(ctx);

// Use gaussians_grad for optimization
optimizer->step(ctx.gaussians_grad, ctx.stream);
```