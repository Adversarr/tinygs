# Strategy Module

The strategy module controls Gaussian densification and pruning during training.

## Files

| File | Description |
|------|-------------|
| `strategy.hpp` | Base class and parameters |
| `default.hpp` | Standard 3DGS strategy |
| `improved.hpp` | Enhanced heuristics |
| `mcmc.hpp` | MCMC-based sampling |

---

## Strategy Types

| Type | Description | Use Case |
|------|-------------|----------|
| `default` | Standard clone/split/prune | General purpose |
| `improved` | Enhanced heuristics | Better quality |
| `mcmc` | MCMC sampling | Recommended for quality |

---

## StrategyParams

Configuration parameters for all strategies:

```cpp
struct StrategyParams {
    // Pruning thresholds
    float pruning_opacity_threshold = 0.005f;   // Remove transparent Gaussians
    float pruning_scale_threshold = 0.1f;        // Remove large Gaussians
    float max_screen_size = 10.0f;               // Remove screen-space large Gaussians
    
    // Growth thresholds (default strategy)
    float duplicate_grad_threshold = 0.0002f;    // Clone if gradient high
    float duplicate_scale_threshold = 0.01f;     // Split if scale large
    
    // Refinement schedule
    bool reset_reset_optimizer = false;          // Reset optimizer on opacity reset
    bool absgrad = false;                        // Use absolute gradient
    int refine_every = 100;                      // Refinement interval
    int start_refine = 500;                      // Start refinement step
    int end_refine = 15000;                      // End refinement step
    int max_num_gaussians = 10000000;            // Maximum Gaussians
    int reset_every = 3000;                      // Opacity reset interval
    uint64_t seed = 42;                          // Random seed
};
```

---

## StrategyBase Interface

```cpp
class StrategyBase {
public:
    StrategyBase(std::shared_ptr<GPUGaussian3d> gaussians,
                 std::shared_ptr<GPUGaussian3d> gaussians_grad,
                 std::shared_ptr<OptimizerBase> optimizer);
    
    /// @brief Execute one strategy step
    void step(const RasterizeContext& ctx);
    
    /// @brief Reset strategy state
    virtual void reset() = 0;
    
    /// @brief Implementation-specific logic
    virtual void step_impl(const RasterizeContext& ctx) = 0;
    
    /// @brief Configuration
    virtual void set_params(const json& config);
    virtual json get_params() const;
    
protected:
    void on_remove(char* kept_flag, int num_kept);
    void on_duplicate(int* indices, int* new_indices, int num_duplications);
    void on_reset(int* indices, int num_reset);
    void on_reset_opacity();
    
    int this_step() const noexcept;
};
```

---

## Default Strategy

Standard 3DGS densification approach:

### Operations

1. **Clone**: Duplicate Gaussians with high position gradient
   - Creates copy at same location
   - Increases local density

2. **Split**: Split large Gaussians into smaller ones
   - Replaces one Gaussian with two smaller ones
   - Improves detail capture

3. **Prune**: Remove undesired Gaussians
   - Low opacity Gaussians
   - Overly large Gaussians
   - Gaussians with high screen-space size

4. **Opacity Reset**: Periodically reset opacity
   - Prevents premature pruning
   - Allows Gaussians to "escape" bad states

```cpp
// Default strategy flow:
// 1. Accumulate gradients over refine_every steps
// 2. Identify Gaussians to clone/split based on gradient
// 3. Identify Gaussians to prune based on opacity/scale
// 4. Apply modifications
// 5. Reset opacity every reset_every steps
```

---

## MCMC Strategy

Markov Chain Monte Carlo based sampling:

- Sample positions for new Gaussians
- Better coverage of scene geometry
- Often produces better quality results

```cpp
// MCMC strategy uses probabilistic sampling
// instead of gradient-based heuristics
```

---

## Creating a Strategy

```cpp
// Via factory
auto strategy = create_strategy("mcmc", gaussians, gradients, optimizer);

// Configure
json params = {
    {"type", "mcmc"},
    {"refine_every", 100},
    {"start_refine", 500},
    {"end_refine", 15000},
    {"max_num_gaussians", 3000000},
    {"pruning_opacity_threshold", 0.005},
    {"reset_every", 3000}
};
strategy->set_params(params);
```

---

## Strategy Step Flow

```
1. Check if refinement should run
   - Step >= start_refine
   - Step <= end_refine
   - Step % refine_every == 0
   
2. Accumulate densification info from rasterizer
   - Position gradients
   - Screen-space sizes
   
3. Identify Gaussians to modify
   - Clone candidates: high gradient, small scale
   - Split candidates: high gradient, large scale
   - Prune candidates: low opacity or oversized
   
4. Apply modifications
   - Update Gaussian data
   - Update optimizer state
   
5. Optionally reset opacity
   - Every reset_every steps
   - Optionally reset optimizer momentum
```

---

## Integration with Optimizer

Strategies must coordinate with optimizers:

```cpp
// When removing Gaussians
on_remove(kept_flags, num_kept);
// -> Calls optimizer->remove()

// When duplicating Gaussians
on_duplicate(indices, new_indices, num_duplications);
// -> Calls optimizer->duplicate()

// When resetting opacity
on_reset_opacity();
// -> Calls optimizer->reset_opacity()
```

---

## Usage Example

```cpp
// Setup
auto strategy = create_strategy("default", gaussians, gradients, optimizer);
strategy->set_params(config["strategy"]);

// In training loop
for (int step = 0; step < max_steps; ++step) {
    // Forward pass
    rasterizer->forward(ctx);
    
    // Loss and backward
    loss->evaluate(loss_ctx, scale);
    rasterizer->backward(ctx);
    
    // Optimization
    optimizer->step(scale, stream);
    
    // Strategy step (calls step_impl internally)
    strategy->step(ctx);
}
```

---

## DensificationInfo

Information stored during rasterization for strategy decisions:

```cpp
struct DensificationInfo {
    float position_gradient_accum;  // Accumulated position gradient
    float denom;                    // Normalization factor
};
```

This data is computed during the backward pass and used by strategies to identify where Gaussians should be added or modified.