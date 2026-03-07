# Architecture Overview

tinygs implements a modular architecture for 3D Gaussian Splatting training and inference. This document describes the high-level design, component relationships, and data flow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Orchestrator                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Training Loop: forward → loss → backward → optimizer → strategy    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         │              │              │              │              │
         ▼              ▼              ▼              ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │DataLoader│  │Rasterizer│  │   Loss   │  │ Optimizer│  │ Strategy │
   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
         │              │              │              │              │
         ▼              ▼              ▼              ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Dataset  │  │GPUGaussian│ │  Image   │  │   Adam   │  │  Default │
   │          │  │    3d    │  │  Buffer  │  │  AdamW   │  │Improved  │
   │  image   │  │ means    │  │          │  │   SGD    │  │   MCMC   │
   │          │  │ opacities│  │          │  │          │  │          │
   └──────────┘  │ scales   │  └──────────┘  └──────────┘  └──────────┘
                 │ rotations│
                 │ sh_coeff │
                 └──────────┘
```

## Core Components

### Orchestrator (`tinygs/orchestrator.hpp`)

The `Orchestrator` class is the central training coordinator that:

- Manages the training loop and state
- Coordinates all other components
- Handles checkpointing and logging
- Manages spherical harmonics degree progression
- Uses dataset-owned training/eval resolution

**Key responsibilities:**
1. Forward pass: Rasterize Gaussians to image
2. Loss computation: Compare rendered image to ground truth
3. Backward pass: Compute gradients for Gaussian parameters
4. Optimization step: Update parameters using optimizer
5. Strategy step: Densify/prune Gaussians periodically

### GPUGaussian3d (`tinygs/core/gpu_gaussian.hpp`)

The core data structure representing 3D Gaussians in GPU memory using Structure of Arrays (SoA) layout:

```cpp
class GPUGaussian3d {
    thrust::device_vector<vec3> m_means;           // 3D positions
    thrust::device_vector<float> m_opacities;      // Opacity values [0,1]
    thrust::device_vector<vec4> m_rotations;       // Quaternion rotations
    thrust::device_vector<vec3> m_scales;          // Scale factors
    thrust::device_vector<vec3> m_sh_coefficient_0;       // DC color
    thrust::device_vector<vec3> m_sh_coefficients_rest;   // SH coefficients 1-15
};
```

### Rasterizer (`tinygs/rasterizer/`)

Renders 3D Gaussians to 2D images using alpha-blending:

**Implementations:**
- `fastgs`: FastGS approximation (faster)
- `fastgs_ours`: Custom FastGS variant
- `fastgs_ours_fp16`: FP16-optimized variant
- `cpu`: CPU reference implementation

**Interface:**
```cpp
class RasterizerBase {
    virtual void forward(const RasterizeContext& params) = 0;
    virtual void backward(RasterizeContext& params) = 0;
};
```

### Optimizer (`tinygs/optim/`)

Updates Gaussian parameters based on gradients:

**Implementations:**
| Type | Description |
|------|-------------|
| `adam` | Full Adam with momentum |
| `adamw` | AdamW with weight decay |
| `sgd` | Stochastic gradient descent |

**Per-parameter learning rates:**
- `means_lr`: Position learning rate
- `shs_lr`: Spherical harmonics learning rate
- `opacities_lr`: Opacity learning rate
- `scales_lr`: Scale learning rate
- `rotations_lr`: Rotation learning rate

### Strategy (`tinygs/strategy/`)

Controls Gaussian densification and pruning:

**Implementations:**
| Type | Description |
|------|-------------|
| `default` | Standard 3DGS (clone, split, prune) |
| `improved` | Enhanced heuristics for better quality |
| `mcmc` | MCMC-based sampling (recommended) |

**Operations:**
1. **Clone**: Duplicate Gaussians with high gradient
2. **Split**: Split large Gaussians into smaller ones
3. **Prune**: Remove low-opacity or oversized Gaussians

### Loss (`tinygs/loss/`)

Computes loss between rendered and target images:

**Losses:**
| Type | Description |
|------|-------------|
| `l1` | L1 (MAE) loss |
| `l2` | L2 (MSE) loss |
| `huber` | Huber loss |
| `fused_ssim` | SSIM+L1 combined (primary) |

**Metrics:**
| Type | Description |
|------|-------------|
| `psnr` | Peak Signal-to-Noise Ratio |

### DataLoader (`tinygs/dataloader/`)

Provides training data batches:

**Implementations:**
| Type | Description |
|------|-------------|
| `simple` | Synchronous single-stream loader |
| `async` | Asynchronous multi-stream with prefetch |

### Dataset (`tinygs/dataset/`)

Loads images and camera parameters:

**Implementations:**
| Type | Description |
|------|-------------|
| `image` | JSON camera + image folder dataset |

## Data Flow

### Training Step

```
1. DataLoader::next()
   ↓
2. Rasterizer::forward(gaussians, camera) → rendered_image
   ↓
3. Loss::evaluate(rendered_image, target_image) → loss, gradients
   ↓
4. Rasterizer::backward(gradients) → gaussian_gradients
   ↓
5. Optimizer::step(gaussian_gradients)
   ↓
6. (periodically) Strategy::step() → modify gaussians
```

### Camera Model

```
Camera Intrinsics (K):
┌──────────────────────┐
│  fx   0   cx   0    │
│   0  fy   cy   0    │
│   0   0    1   0    │
│   0   0    0   1    │
└──────────────────────┘

Camera Extrinsics:
- Rotation: quaternion (w, x, y, z)
- Translation: vec3 (x, y, z)
- World-to-Camera matrix: mat4x4
```

## Memory Management

### GPU Memory Arena

The library uses a memory arena pattern for efficient allocation:

```cpp
class GPUMemoryArena {
    // Pooled allocations for intermediate buffers
    // Reused across forward/backward passes
};
```

### Image Buffer Layout

Images use AoSoA (Array of Structures of Arrays) tiling for cache efficiency:

```cpp
// 8x8 tiles for coalesced memory access
constexpr uint32_t kImageTile = 8;

// Tiled linear index: tile_idx * 64 + intra_tile_offset
uint32_t idx = get_linear_index(i, j, width);
```

## Factory Pattern

All major components use the factory pattern for creation:

```cpp
// Example: Creating components from config
auto rasterizer = create_rasterizer("fastgs");
auto optimizer = create_optimizer("adam", gaussians, gradients);
auto strategy = create_strategy("mcmc", gaussians, gradients, optimizer);
auto dataloader = create_dataloader("async", dataset);
auto dataset = create_dataset("image");
auto loss = create_loss("fused_ssim");
```

## Configuration System

All components support JSON configuration via `set_params()`/`get_params()`:

```json
{
  "rasterizer": { "type": "fastgs" },
   "optimizer": { "type": "adam", "means_lr": 0.00016, "decouple_decay": true },
  "strategy": { "type": "mcmc", "refine_every": 100 }
}
```

## Queue Management

Execution queues are explicitly passed for asynchronous operations:

```cpp
void train_step() {
    // Major queue for training operations
    const BackendQueue* queue = m_major_queue.get();
    
    // Forward pass
    rasterizer->forward(ctx);
    
    // Loss computation (async)
    loss->evaluate(loss_ctx, scale);
    
    // Backward pass
    rasterizer->backward(ctx);
    
    // Optimization step
    optimizer->step(scale, queue);
}
```