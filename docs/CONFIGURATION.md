# Configuration

tinygs uses JSON configuration files to define all training parameters. This document describes the configuration format and available options.

## Configuration Structure

A complete configuration file contains:

```json
{
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

---

## Dataset Configuration

### dataset

Defines the training dataset:

```json
{
  "dataset": {
    "type": "image",
    "root_path": "outputs/garden/train/",
    "extension": "png",
    "resolution": -1,
    "resolution_scale": 1.0
  }
}
```

**Common parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | string | Dataset type: `image` |
| `root_path` | string | Dataset root containing `cameras.json`, `poses.json`, `images/` |
| `extension` | string | Image extension: `png`, `jpg` |
| `resolution` | int | Resolution mode: `{1,2,4,8}` divisor, `-1` auto(1600px cap), `>0` target width |
| `resolution_scale` | float | Additional scale divisor applied on top of `resolution` |

### test_dataset (optional)

Same structure as `dataset`, for evaluation:

```json
{
  "test_dataset": {
    "type": "image",
    "root_path": "outputs/garden/val/",
    "extension": "png",
    "resolution": -1,
    "resolution_scale": 1.0
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
    "type": "adam",
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
| `type` | string | - | `adam` |
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
| `decouple_decay` | bool | false | Enable decoupled weight decay (AdamW mode) |

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
| `type` | string | - | `default`, `improved`, `mcmc`, `fastgs` |
| `refine_every` | int | 100 | Refinement interval |
| `start_refine` | int | 500 | Start refinement step |
| `end_refine` | int | 15000 | End refinement step |
| `max_num_gaussians` | int | 10M | Maximum Gaussians |
| `pruning_opacity_threshold` | float | 0.005 | Opacity prune threshold |
| `pruning_scale_threshold` | float | 0.1 | Scale prune threshold |
| `reset_every` | int | 3000 | Opacity reset interval |
| `absgrad` | bool | false | Use absolute gradient |
| `seed` | int | 42 | Random seed |

### FastGS-specific strategy parameters

When `strategy.type = "fastgs"`, the following additional fields are supported:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_thresh` | float | `0.1` | Metric-map threshold. With normalized L1 enabled, threshold is in `[0,1]`. |
| `normalize_metric_l1` | bool | `true` | Min-max normalize per-pixel L1 before thresholding (Python parity behavior). |
| `metric_num_cameras` | int | `10` | Number of sampled cameras for FastGS metric scoring. |
| `sample_cameras_without_replacement` | bool | `true` | Camera sampling mode for scoring; Python parity uses without replacement. |
| `photometric_l1_weight` | float | `0.8` | Weight for L1 term in photometric score (auto-normalized with SSIM weight). |
| `photometric_ssim_weight` | float | `0.2` | Weight for SSIM-loss term `(1 - SSIM)` in photometric score (auto-normalized). |
| `sanitize_nan_gradients` | bool | `true` | Replaces non-finite gradient statistics with `0` during densification classification. |
| `importance_threshold` | float | `5.0` | Minimum FastGS importance score required for clone/split candidacy. |
| `prune_budget_ratio` | float | `0.5` | Fraction of standard prune candidates removed each densification step. |
| `use_multinomial_pruning` | bool | `true` | Uses stochastic multinomial (without replacement) prune sampling (Python parity). |
| `prune_degenerate_rotation` | bool | `false` | If enabled, additionally prunes Gaussians with degenerate rotations. |
| `final_prune_score_threshold` | float | `0.9` | Hard prune threshold on normalized FastGS pruning score. |
| `final_prune_opacity_threshold` | float | `0.1` | Hard prune threshold on activated opacity. |
| `final_prune_start` | int | `18000` | First step to run final prune checks. |
| `final_prune_end` | int | `27000` | Last step to run final prune checks. |
| `final_prune_every` | int | `3000` | Step interval for final prune checks. |
| `opacity_reset_value` | float | `0.8` | Post-densification opacity clamp maximum. |

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
| `start_pose_opt` | int | 500 | Step to start pose optimization |

---

## Example Configuration

Minimal working configuration:

```json
{
  "dataset": {
    "type": "image",
    "root_path": "outputs/scene/train/",
    "extension": "png",
    "resolution": -1,
    "resolution_scale": 1.0
  },
  "dataloader": { "type": "async" },
  "test_dataset": {
    "type": "image",
    "root_path": "outputs/scene/val/",
    "extension": "png",
    "resolution": -1,
    "resolution_scale": 1.0
  },
  "initializer": { "type": "knn" },
  "rasterizer": { "type": "fastgs" },
  "optimizer": { "type": "adam" },
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