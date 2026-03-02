# Optimizer Module

The optimizer module provides gradient-based optimization algorithms for updating Gaussian parameters.

## Files

| File | Description |
|------|-------------|
| `optim.hpp` | Base class and shared parameters |
| `adam.hpp` | Adam optimizer |
| `adamw.hpp` | AdamW with weight decay |
| `sgd.hpp` | Stochastic gradient descent |
| `lr_scheduler.hpp` | Learning rate schedulers |

---

## Optimizer Types

| Type | Description | Use Case |
|------|-------------|----------|
| `adam` | Adam optimizer | Default training optimizer |
| `adamw` | AdamW with weight decay | When regularization needed |
| `sgd` | Stochastic gradient descent | Simple baseline |

---

## GaussianOptimizationParams

Shared parameters for all optimizers:

```cpp
struct GaussianOptimizationParams {
    // Gradient clipping
    float max_grad_1 = 1.0f;       // L1 clipping threshold
    bool skip_zero_grad = false;    // Skip zero gradients
    
    // Per-parameter learning rates
    float means_lr = 1.6e-4f;       // Position LR
    float shs_lr = 2.5e-3f;         // SH coefficients LR
    float opacities_lr = 5.0e-2f;   // Opacity LR
    float scales_lr = 5.0e-3f;      // Scale LR
    float rotations_lr = 1.0e-3f;   // Rotation LR
    
    // L1 regularization
    float opacities_l1 = 0.0f;      // Opacity regularization
    float scales_l1 = 0.0f;         // Scale regularization
};
```

---

## OptimizerBase Interface

```cpp
class OptimizerBase {
public:
    OptimizerBase(std::shared_ptr<GPUGaussian3d> gaussians,
                  std::shared_ptr<GPUGaussian3d> gaussians_grad);
    
    /// @brief Perform one optimization step
    virtual void step(float scale, cudaStream_t stream) = 0;
    
    /// @brief Reset all optimizer state
    virtual void reset() = 0;
    
    /// @brief Handle Gaussian removal
    virtual void remove(char* kept_flag, int num_kept) {}
    
    /// @brief Handle Gaussian duplication
    virtual void duplicate(int* indices, int* new_indices, int num_duplicate) {}
    
    /// @brief Reset specific Gaussians
    virtual void reset(int* indices, int num_reset) = 0;
    
    /// @brief Reorder Gaussians
    virtual void reorder(uint* indices) = 0;
    
    /// @brief Reset opacity-related state
    virtual void reset_opacity() = 0;
    
    /// @brief Learning rate management
    void set_lr(float new_lr);
    float get_lr() const;
    
    /// @brief Configuration
    virtual void set_params(const json& config);
    virtual json get_params() const;
};
```

---

## AdamW Optimizer

Full Adam implementation with momentum storage:

```cpp
struct AdamWParameters {
    float beta1 = 0.9f;            // First moment decay
    float beta2 = 0.999f;          // Second moment decay
    float epsilon = 1e-8f;         // Numerical stability
    bool enable_adabound = false;  // AdaBound extension
};

class AdamW : public OptimizerBase {
    // Stores first and second moments for all parameters
    thrust::device_vector<vec3> m_means_first_second;
    thrust::device_vector<float> m_opacities_first_second;
    thrust::device_vector<vec4> m_rotations_first_second;
    thrust::device_vector<vec3> m_scales_first_second;
    thrust::device_vector<vec3> m_sh_coefficient_0_first_second;
    thrust::device_vector<vec3> m_sh_coefficients_rest_first_second;
    thrust::device_vector<uint32_t> m_gaussian_steps;
};
```

---

## Learning Rate Schedulers

### LrSchedulerBase

```cpp
class LrSchedulerBase {
public:
    LrSchedulerBase(const std::shared_ptr<OptimizerBase>& optimizer, float initial_lr);
    
    /// @brief Step and return new LR
    virtual float step() = 0;
    
    /// @brief Reset to initial state
    virtual void reset() = 0;
    
    float get_lr() const;
    
    virtual json get_params() const = 0;
    virtual void set_params(const json& params) = 0;
};
```

### ConstantLR

```cpp
// Always returns initial learning rate
class ConstantLR : public LrSchedulerBase {
    float step() override;  // Returns constant LR
};
```

### ExponentialLR

```cpp
// lr = initial_lr * (decay_rate ^ step_count)
class ExponentialLR : public LrSchedulerBase {
    float m_initial_lr = 1.0f;
    float m_decay_rate = 0.999769f;
    int m_step_count = 0;
    
    float step() override;
};
```

---

## Creating an Optimizer

```cpp
// Via factory
auto optimizer = create_optimizer("adamw", gaussians, gradients);

// Configure
json params = {
    {"type", "adamw"},
    {"means_lr", 0.00016},
    {"shs_lr", 0.0025},
    {"opacities_lr", 0.05},
    {"scales_lr", 0.005},
    {"rotations_lr", 0.001}
};
optimizer->set_params(params);
```

---

## Creating a Scheduler

```cpp
// Via factory
auto scheduler = create_lr_scheduler("exponential", optimizer);

// Or directly
auto scheduler = std::make_shared<ExponentialLR>(optimizer, 1.0f, 0.999869f);
```

---

## Usage Example

```cpp
// Setup
auto optimizer = create_optimizer("adam", gaussians, gradients);
auto scheduler = create_lr_scheduler("exponential", optimizer);

// Training loop
for (int step = 0; step < max_steps; ++step) {
    // Forward pass
    rasterizer->forward(ctx);
    
    // Compute loss
    loss->evaluate(loss_ctx, scale);
    
    // Backward pass
    rasterizer->backward(ctx);
    
    // Update learning rate
    scheduler->step();
    
    // Optimization step
    optimizer->step(scale, stream);
    
    // Periodically densify
    if (step % refine_every == 0) {
        strategy->step(ctx);
    }
}
```

---

## Strategy Integration

Optimizers must handle Gaussian modifications from strategies:

```cpp
// When Gaussians are removed
optimizer->remove(kept_flags, num_kept);

// When Gaussians are duplicated
optimizer->duplicate(indices, new_indices, num_duplicate);

// When Gaussians are reordered
optimizer->reorder(reorder_indices);

// When opacity is reset
optimizer->reset_opacity();
```