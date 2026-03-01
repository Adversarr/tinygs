# Loss Module

The loss module provides loss functions and metrics for training and evaluation.

## Files

| File | Description |
|------|-------------|
| `loss.hpp` | Base classes for loss and metrics |
| `l1.hpp` | L1 (MAE) loss |
| `l2.hpp` | L2 (MSE) loss |
| `huber.hpp` | Huber loss |
| `fused_ssim.hpp` | Fused SSIM+L1 loss |
| `psnr.hpp` | PSNR metric |

---

## Loss Functions

| Type | Description | Use Case |
|------|-------------|----------|
| `l1` | L1 (Mean Absolute Error) | Simple, robust |
| `l2` | L2 (Mean Squared Error) | Penalizes outliers |
| `huber` | Huber loss | Robust to outliers |
| `fused_ssim` | SSIM+L1 combined | Primary training loss (recommended) |

---

## Metrics

| Type | Description |
|------|-------------|
| `psnr` | Peak Signal-to-Noise Ratio |

---

## LossContext

```cpp
struct LossContext {
    Image pred;      // Predicted image (from rasterizer)
    Image target;    // Ground truth image
    Image loss;      // Per-pixel loss output
    Image grad;      // Gradient for backpropagation
    cudaStream_t stream = nullptr;
};
```

---

## LossBase Interface

```cpp
class LossBase {
public:
    virtual ~LossBase() = default;
    
    /// @brief Compute loss and accumulate gradients
    /// @param ctx Loss context with pred, target, loss, grad
    /// @param scale Scaling factor for loss and gradient
    virtual void evaluate(LossContext ctx, float scale) = 0;
    
    /// @brief Get loss function name
    virtual std::string name() const = 0;
};
```

---

## MetricBase Interface

```cpp
class MetricBase {
public:
    virtual ~MetricBase() = default;
    
    /// @brief Evaluate metric (no gradients)
    /// @return Metric value
    virtual float evaluate(Image pred, Image target) = 0;
    
    /// @brief Get metric name
    virtual std::string name() const = 0;
};
```

---

## L1 Loss

Mean Absolute Error:

```cpp
// L1 = (1/N) * sum(|pred - target|)
class L1Loss : public LossBase {
    void evaluate(LossContext ctx, float scale) override;
    std::string name() const override { return "l1"; }
};
```

---

## L2 Loss

Mean Squared Error:

```cpp
// L2 = (1/N) * sum((pred - target)^2)
class L2Loss : public LossBase {
    void evaluate(LossContext ctx, float scale) override;
    std::string name() const override { return "l2"; }
};
```

---

## Huber Loss

Combines L1 and L2 for robustness:

```cpp
// Huber = {
//   0.5 * (pred - target)^2     if |pred - target| <= delta
//   delta * (|pred - target| - 0.5 * delta)  otherwise
// }
class HuberLoss : public LossBase {
    void evaluate(LossContext ctx, float scale) override;
    std::string name() const override { return "huber"; }
};
```

---

## Fused SSIM Loss

Combined SSIM + L1 loss (recommended for training):

```cpp
// Loss = lambda * (1 - SSIM) + (1 - lambda) * L1
// where lambda is typically 0.2
class FusedSSIMLoss : public LossBase {
    void evaluate(LossContext ctx, float scale) override;
    std::string name() const override { return "fused_ssim"; }
};
```

**Why use fused SSIM?**
- SSIM captures perceptual similarity
- L1 provides pixel-level supervision
- Combined loss is more robust

---

## PSNR Metric

Peak Signal-to-Noise Ratio:

```cpp
// PSNR = 10 * log10(MAX^2 / MSE)
// where MAX = 1.0 for normalized images
class PSNRMetric : public MetricBase {
    float evaluate(Image pred, Image target) override;
    std::string name() const override { return "psnr"; }
};
```

**Interpretation:**
- Higher is better
- Typical values: 20-40 dB
- > 30 dB is considered good quality

---

## Creating Losses and Metrics

```cpp
// Create loss
auto loss = create_loss("fused_ssim");

// Create metric
auto metric = create_metric("psnr");
```

---

## Combined Loss

Multiple losses can be combined with weights:

```cpp
// In config_train or orchestrator
orchestrator->add_loss(std::shared_ptr<LossBase>(create_loss("l1")), 0.8f);
orchestrator->add_loss(std::shared_ptr<LossBase>(create_loss("fused_ssim")), 0.2f);

// Total loss = 0.8 * L1 + 0.2 * (1 - SSIM)
```

---

## Configuration

### Losses (JSON)

```json
{
    "losses": [
        {"type": "l1", "weight": 0.8},
        {"type": "fused_ssim", "weight": 0.2}
    ]
}
```

### Metrics (JSON)

```json
{
    "metrics": ["psnr"]
}
```

---

## Usage Example

```cpp
// Setup
auto l1_loss = create_loss("l1");
auto ssim_loss = create_loss("fused_ssim");
auto psnr = create_metric("psnr");

// Training step
LossContext ctx;
ctx.pred = rendered_image;
ctx.target = target_image;
ctx.loss = loss_buffer;
ctx.grad = grad_buffer;
ctx.stream = cuda_stream;

// Evaluate losses
l1_loss->evaluate(ctx, 0.8f);
ssim_loss->evaluate(ctx, 0.2f);

// Use ctx.grad for backpropagation

// Evaluation
float psnr_value = psnr->evaluate(rendered_image, target_image);
log_info("PSNR: {:.2f} dB", psnr_value);
```

---

## GPU Implementation

Losses are implemented as CUDA kernels:

```cpp
// Example: L1 kernel
__global__ void l1_loss_kernel(
    const float* pred,
    const float* target,
    float* loss,
    float* grad,
    size_t n,
    float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float diff = pred[idx] - target[idx];
    loss[idx] = abs(diff) * scale;
    grad[idx] = (diff > 0 ? 1.0f : -1.0f) * scale;
}
```

---

## Gradient Accumulation

When using multiple losses, gradients accumulate:

```cpp
// Loss 1: adds to grad
l1_loss->evaluate(ctx, 0.8f);

// Loss 2: adds to grad (accumulates)
ssim_loss->evaluate(ctx, 0.2f);

// Final grad = 0.8 * grad_l1 + 0.2 * grad_ssim
```