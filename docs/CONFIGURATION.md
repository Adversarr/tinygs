# Configuration

tinygs uses JSON configuration files to define all training parameters. This document describes the configuration format and available options.

## Configuration Structure

A complete configuration file contains:

```json
{
  "input_pc_file": "path/to/points.ply",
  "dataset": { ... },
  "dataloader": { ... },
  "test_dataset": { ... },
  "test_dataloader": { ... },
  "initializer": { ... },
  "rasterizer": { ... },
  "optimizer": { ... },
  "lr_scheduler": { ... },
  "pose_opt": { ... },
  "losses": [ ... ],
  "metrics": [ ... ],
  "strategy": { ... },
  "trainer": { ... }
}
```

## Required Fields

### input_pc_file

Path to the initial point cloud file (PLY or TXT format):

```json
{
  "input_pc_file": "outputs/garden/points.ply"
}
```

---

## Dataset Configuration

### dataset

Defines the training dataset:

```json
{
  "dataset": {
    "type": "png_folder",
    "folder_path": "outputs/garden/images/",
    "extrinsics_file_path": "outputs/garden/extri.txt",
    "intrinsics_file_path": "outputs/garden/intri.txt",
    "interpolate": true,
    "undistortion": false,
    "extension": "png"
  }
}
```

**Common parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | Dataset type: `png_folder`, `video` |
| `folder_path` | string | Path to image folder |
| `extrinsics_file_path` | string | Path to extrinsics file |
| `intrinsics_file_path` | string | Path to intrinsics file |
| `interpolate` | bool | Interpolate camera poses |
| `undistortion` | bool | Apply undistortion |
| `extension` | string | Image extension: `png`, `jpg` |

**Video dataset:**

```json
{
  "dataset": {
    "type": "video",
    "video_file_path": "path/to/video.mp4",
    "video_info_path": "path/to/videoInfo.txt"
  }
}
```

### test_dataset (optional)

Same structure as `dataset`, for evaluation:

```json
{
  "test_dataset": {
    "type": "png_folder",
    "folder_path": "outputs/garden/test_images/",
    ...
  }
}
```

---

## DataLoader Configuration

### dataloader

```json
{
  "dataloader": {
    "type": "async",
    "data_type": "float16"
  }
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | `simple` or `async` (recommended) |
| `data_type` | string | `float32` or `float16` |

---

## Initialization Configuration

### initializer

```json
{
  "initializer": {
    "type": "knn",
    "num_neighbors": 8,
    "default_distance": 0.01,
    "init_opacity": 0.1,
    "init_scaling": 0.6,
    "sh_degree": 3,
    "enable_radius_outlier_removal": false,
    "radius": 0.05,
    "nb_points": 16,
    "min_distance": 1.0e-7
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | `knn` | `knn` or `random` |
| `num_neighbors` | int | 8 | KNN neighbors for scale estimation |
| `default_distance` | float | 0.01 | Default scale if KNN fails |
| `init_opacity` | float | 0.1 | Initial opacity |
| `init_scaling` | float | 0.6 | Scale multiplier |
| `sh_degree` | int | 3 | Max SH degree |
| `enable_radius_outlier_removal` | bool | false | Remove outliers |
| `radius` | float | 0.05 | Outlier removal radius |
| `nb_points` | int | 16 | Min points in radius |

---

## Rasterizer Configuration

### rasterizer

```json
{
  "rasterizer": {
    "type": "fastgs"
  }
}
```

| Type | Description |
|------|-------------|
| `default` | Reference 3DGS implementation |
| `fastgs` | FastGS (recommended) |
| `gsplat` | GSplat implementation |

---

## Optimizer Configuration

### optimizer

```json
{
  "optimizer": {
    "type": "simple_adam",
    "means_lr": 0.00016,
    "shs_lr": 0.0025,
    "opacities_lr": 0.05,
    "scales_lr": 0.005,
    "rotations_lr": 0.001,
    "max_grad_1": 1.0,
    "skip_zero_grad": true,
    "opacities_l1": 0.01,
    "scales_l1": 0.01,
    "epsilon": 1.0e-8
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | - | `simple_adam`, `adam`, `adamw`, `lion`, `lamb`, `adan`, `sgd` |
| `means_lr` | float | 1.6e-4 | Position learning rate |
| `shs_lr` | float | 2.5e-3 | SH coefficients learning rate |
| `opacities_lr` | float | 5.0e-2 | Opacity learning rate |
| `scales_lr` | float | 5.0e-3 | Scale learning rate |
| `rotations_lr` | float | 1.0e-3 | Rotation learning rate |
| `max_grad_1` | float | 1.0 | Gradient clipping threshold |
| `skip_zero_grad` | bool | true | Skip zero gradients |
| `opacities_l1` | float | 0.0 | L1 regularization for opacity |
| `scales_l1` | float | 0.0 | L1 regularization for scale |
| `epsilon` | float | 1e-8 | Adam epsilon |

---

## Learning Rate Scheduler Configuration

### lr_scheduler

```json
{
  "lr_scheduler": {
    "type": "exponential",
    "initial_lr": 1.0,
    "decay_rate": 0.999869
  }
}
```

| Type | Parameters |
|------|------------|
| `exponential` | `initial_lr`, `decay_rate` |
| `cosine` | `initial_lr`, `final_lr` |
| `step` | `initial_lr`, `decay_steps`, `decay_rate` |

---

## Pose Optimization Configuration

### pose_opt

```json
{
  "pose_opt": {
    "type": "adamw",
    "lr": 0.0001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.1
  }
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | `none`, `adamw`, `sgdm` |
| `lr` | float | Learning rate |
| `beta1` | float | Adam beta1 |
| `beta2` | float | Adam beta2 |
| `weight_decay` | float | Weight decay |

---

## Loss Configuration

### losses

Array of loss functions with weights:

```json
{
  "losses": [
    { "type": "l1", "weight": 0.8 },
    { "type": "fused_ssim", "weight": 0.2 }
  ]
}
```

| Type | Description |
|------|-------------|
| `l1` | L1 (MAE) loss |
| `l2` | L2 (MSE) loss |
| `huber` | Huber loss |
| `fused_ssim` | SSIM+L1 combined (recommended) |

---

## Metrics Configuration

### metrics

Array of metric names:

```json
{
  "metrics": ["psnr"]
}
```

| Type | Description |
|------|-------------|
| `psnr` | Peak Signal-to-Noise Ratio |

---

## Strategy Configuration

### strategy

```json
{
  "strategy": {
    "type": "mcmc",
    "refine_every": 100,
    "start_refine": 500,
    "end_refine": 15000,
    "max_num_gaussians": 3000000,
    "pruning_opacity_threshold": 0.005,
    "pruning_scale_threshold": 0.1,
    "max_screen_size": 10,
    "reset_every": 3000,
    "reset_reset_optimizer": false,
    "seed": 42,
    "absgrad": true,
    "duplicate_grad_threshold": 0.0008,
    "duplicate_scale_threshold": 0.005
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | - | `default`, `improved`, `mcmc` |
| `refine_every` | int | 100 | Refinement interval |
| `start_refine` | int | 500 | Start refinement step |
| `end_refine` | int | 15000 | End refinement step |
| `max_num_gaussians` | int | 10M | Maximum Gaussians |
| `pruning_opacity_threshold` | float | 0.005 | Opacity prune threshold |
| `pruning_scale_threshold` | float | 0.1 | Scale prune threshold |
| `reset_every` | int | 3000 | Opacity reset interval |
| `absgrad` | bool | false | Use absolute gradient |
| `seed` | int | 42 | Random seed |

---

## Trainer Configuration

### trainer

```json
{
  "trainer": {
    "max_steps": 30000,
    "max_seconds": 240,
    "log_interval": 100,
    "checkpoint_interval": 1000,
    "sh_degree_interval": 1000,
    "max_sh_degree": 3,
    "near_plane": 0.01,
    "far_plane": 100.0,
    "grad_scaler": 128.0,
    "train_data_type": "float16",
    "eval_data_type": "float16",
    "out_dir": "outputs",
    "test_steps": [7000, 30000],
    "enable_progressive_resolution": false,
    "resolution": -1,
    "resolution_scale": 1.0,
    "start_pose_opt": 500
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | int | 30000 | Maximum training steps |
| `max_seconds` | int | 0 | Time limit (0 = disabled) |
| `log_interval` | int | 100 | Logging interval |
| `checkpoint_interval` | int | 1000 | Checkpoint interval |
| `sh_degree_interval` | int | 1000 | SH degree increase interval |
| `max_sh_degree` | int | 3 | Maximum SH degree |
| `near_plane` | float | 0.01 | Near plane distance |
| `far_plane` | float | 100.0 | Far plane distance |
| `grad_scaler` | float | 1.0 | Gradient scaling (128.0 for fp16) |
| `train_data_type` | string | float32 | `float32` or `float16` |
| `eval_data_type` | string | float32 | `float32` or `float16` |
| `out_dir` | string | "" | Output directory |
| `test_steps` | array | [7000, 30000] | Steps to run evaluation |
| `resolution` | int | -1 | Resolution mode: {1,2,4,8}=divisor, -1=auto (cap 1600px), >0=target width |
| `resolution_scale` | float | 1.0 | Additional resolution scale factor (divisor) |
| `start_pose_opt` | int | 500 | Step to start pose optimization |

---

## Example Configuration

Minimal working configuration:

```json
{
  "input_pc_file": "outputs/scene/points.ply",
  "dataset": {
    "type": "png_folder",
    "folder_path": "outputs/scene/images/",
    "extrinsics_file_path": "outputs/scene/extri.txt",
    "intrinsics_file_path": "outputs/scene/intri.txt"
  },
  "dataloader": { "type": "async" },
  "initializer": { "type": "knn" },
  "rasterizer": { "type": "fastgs" },
  "optimizer": { "type": "simple_adam" },
  "lr_scheduler": { "type": "exponential" },
  "pose_opt": { "type": "adamw" },
  "losses": [
    { "type": "l1", "weight": 0.8 },
    { "type": "fused_ssim", "weight": 0.2 }
  ],
  "metrics": ["psnr"],
  "strategy": { "type": "mcmc" },
  "trainer": {
    "max_steps": 30000,
    "out_dir": "outputs"
  }
}
```