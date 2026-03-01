# Pose Optimization Module

The pose optimization module handles camera pose refinement during training.

## Files

| File | Description |
|------|-------------|
| `pose_opt.hpp` | Base class interface |
| `adamw.hpp` | AdamW-based pose optimization |
| `sgdm.hpp` | SGD with momentum |
| `none.hpp` | No optimization (fixed poses) |

---

## Pose Optimizer Types

| Type | Description | Use Case |
|------|-------------|----------|
| `none` | No pose optimization | Fixed camera poses |
| `adamw` | AdamW optimizer | Adaptive learning |
| `sgdm` | SGD with momentum | Simple baseline |

---

## PoseOptBase Interface

```cpp
class PoseOptBase {
public:
    virtual ~PoseOptBase() = default;
    
    /// @brief Query optimized pose matrix
    /// @param timestamp Frame timestamp
    /// @param world_to_camera Original pose matrix
    /// @return Optimized pose matrix
    virtual mat4x4 query(uuid_t timestamp, mat4x4 world_to_camera) = 0;
    
    /// @brief Update pose from gradient
    /// @param timestamp Frame timestamp
    /// @param grad_pose Gradient of pose matrix
    /// @param step_size Learning rate scale
    virtual void update(uuid_t timestamp, const mat4x4& grad_pose, float step_size) = 0;
    
    /// @brief Configuration
    virtual json get_params() const noexcept = 0;
    virtual void set_params(const json& params) = 0;
};
```

---

## AdamW Pose Optimizer

AdamW optimizer for camera poses:

### Configuration

```json
{
    "type": "adamw",
    "lr": 0.0001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.1
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.0001 | Learning rate |
| `beta1` | float | 0.9 | First moment decay |
| `beta2` | float | 0.999 | Second moment decay |
| `epsilon` | float | 1e-8 | Numerical stability |
| `weight_decay` | float | 0.1 | Weight decay |

### Implementation

```cpp
class AdamWPoseOpt : public PoseOptBase {
    // Per-timestamp pose deltas and momentum
    std::unordered_map<uuid_t, PoseDelta> m_deltas;
    std::unordered_map<uuid_t, PoseMomentum> m_momentum;
    
    mat4x4 query(uuid_t timestamp, mat4x4 w2c) override;
    void update(uuid_t timestamp, const mat4x4& grad, float step_size) override;
};
```

---

## SGD with Momentum

Simple SGD with momentum for pose optimization:

### Configuration

```json
{
    "type": "sgdm",
    "lr": 0.0001,
    "momentum": 0.9
}
```

---

## No Optimization

Disables pose optimization:

```json
{
    "type": "none"
}
```

---

## Creating a Pose Optimizer

```cpp
// Via factory
auto pose_opt = create_pose_opt("adamw");

// Configure
pose_opt->set_params({
    {"lr", 0.0001},
    {"beta1", 0.9},
    {"beta2", 0.999}
});
```

---

## Usage in Training

Pose optimization is integrated into the training loop:

```cpp
// In Orchestrator
void train_step() {
    // Get current frame
    auto batch = m_dataloader->next();
    uuid_t timestamp = batch.input.timestamp;
    
    // Query optimized pose
    mat4x4 optimized_w2c = m_pose_opt->query(timestamp, batch.input.w2c);
    
    // Use optimized pose for rendering
    ctx.fwd_input.w2c = optimized_w2c;
    m_rasterizer->forward(ctx);
    
    // ... loss and backward ...
    
    // Get pose gradient from backward pass
    if (ctx.grad_input.has_value()) {
        mat4x4 pose_grad = ctx.grad_input.value().w2c;
        
        // Update pose
        float step_size = m_lr_scheduler->get_lr();
        m_pose_opt->update(timestamp, pose_grad, step_size);
    }
}
```

---

## Pose Parameterization

Poses can be optimized in different parameterizations:

### SE(3) Parameterization

Direct optimization of rotation and translation:

```cpp
// Delta rotation: quaternion
// Delta translation: vec3
struct PoseDelta {
    quat delta_rotation;
    vec3 delta_translation;
};
```

### Axis-Angle Parameterization

Rotation as axis-angle vector:

```cpp
// Rotation: vec3 (axis * angle)
// Translation: vec3
```

---

## Starting Pose Optimization

Pose optimization typically starts after initial training:

```cpp
// In config
{
    "trainer": {
        "start_pose_opt": 500  // Start at step 500
    }
}
```

This allows the Gaussians to stabilize before refining poses.

---

## Gradient Flow

```
1. Forward: Use optimized pose
   w2c_opt = pose_opt->query(timestamp, w2c_orig)
   
2. Backward: Compute pose gradient
   d_loss / d_w2c
   
3. Update: Apply gradient to pose
   pose_opt->update(timestamp, d_w2c, step_size)
```

---

## Regularization

Pose optimization benefits from regularization:

- **Weight decay**: Prevents large pose changes
- **Temporal smoothing**: Encourages smooth pose trajectories
- **Pose priors**: Encourages poses to stay near initial estimates

---

## Usage Example

```cpp
// Setup
auto pose_opt = create_pose_opt("adamw");
pose_opt->set_params({
    {"lr", 0.0001},
    {"weight_decay", 0.1}
});

// During training
for (int step = 0; step < max_steps; ++step) {
    auto batch = dataloader->next();
    
    // Apply pose optimization
    if (step >= start_pose_opt) {
        batch.input.w2c = pose_opt->query(
            batch.input.timestamp,
            batch.input.w2c
        );
    }
    
    // ... training step ...
    
    // Update pose
    if (step >= start_pose_opt && pose_grad_available) {
        pose_opt->update(
            batch.input.timestamp,
            pose_gradient,
            current_lr
        );
    }
}
```

---

## Notes

- Pose optimization requires `prepare_input_gradients = true` in RasterizeContext
- Small learning rates (1e-4 to 1e-5) are typically best
- Start after initial convergence (500+ steps)
- Regularization helps prevent drift