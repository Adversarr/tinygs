# Random Module

The random module provides random number generation utilities for probabilistic operations.

## Files

| File | Description |
|------|-------------|
| `multinomial.hpp` | Multinomial sampling |
| `pcg32.hpp` | PCG random number generator |

---

## Multinomial Sampling

Sample from a categorical distribution based on weights.

### CUDA Implementation (With Replacement)

```cpp
/// @brief Multinomial sampling with replacement on GPU
/// @param d_weights Weight array on GPU (non-negative)
/// @param K Number of categories
/// @param num_samples Number of samples to draw
/// @param seed Random seed
/// @param queue Execution queue
/// @return Buffer of sampled indices
GPUBuffer<int> multinomial_cuda_with_replacement(
    const float* d_weights,
    int K,
    int num_samples,
    int seed,
    cudaStream_t stream = 0
);
```

### CUDA Implementation (Without Replacement)

```cpp
/// @brief Multinomial sampling without replacement on GPU
/// Uses Efraimidis-Spirakis PPS sampling
/// @param d_weights Weight array on GPU
/// @param K Number of categories
/// @param num_samples Number of samples (must be <= K)
/// @param seed Random seed
/// @param queue Execution queue
/// @return Buffer of sampled indices
GPUBuffer<int> multinomial_cuda_cpu_without_replacement(
    const float* d_weights,
    int K,
    int num_samples,
    int seed,
    cudaStream_t stream = 0
);
```

### CPU Implementations

```cpp
/// @brief CPU multinomial with replacement
std::vector<int> multinomial_cpu_with_replacement(
    const float* weights,
    int K,
    int num_samples,
    int seed
);

/// @brief CPU multinomial without replacement
std::vector<int> multinomial_cpu_without_replacement(
    const float* weights,
    int K,
    int num_samples,
    int seed
);
```

---

## Algorithm Details

### With Replacement

Standard categorical sampling:
1. Normalize weights to probabilities
2. Compute cumulative distribution
3. Sample uniform random number
4. Binary search to find category

### Without Replacement (Efraimidis-Spirakis)

Priority-based sampling:
1. For each category i, generate key_i = u_i^(1/w_i) where u_i is uniform [0,1]
2. Select top-num_samples categories by key value
3. This gives PPS (Probability Proportional to Size) sampling

---

## Usage in Gaussian Splatting

Multinomial sampling is used in densification strategies:

### MCMC Strategy

```cpp
// Sample Gaussians for splitting
auto weights = compute_split_weights(gaussians);
auto selected = multinomial_cuda_cpu_without_replacement(
    weights.data(),
    gaussians.size(),
    num_to_split,
    seed,
    stream
);

// selected contains indices of Gaussians to split
```

### Clone Sampling

```cpp
// Sample positions for new Gaussians
auto weights = compute_clone_weights();
auto clone_indices = multinomial_cuda_with_replacement(
    weights.data(),
    weights.size(),
    num_clones,
    seed,
    stream
);
```

---

## PCG32 Random Generator

High-quality random number generator.

```cpp
class pcg32 {
public:
    using result_type = uint32_t;
    
    pcg32(uint64_t seed = 0);
    
    uint32_t operator()();
    
    void seed(uint64_t seed);
    
    static constexpr uint32_t min();
    static constexpr uint32_t max();
};
```

### Usage

```cpp
pcg32 rng(42);  // Seed

// Generate random numbers
uint32_t r1 = rng();  // Random uint32_t
float r2 = float(rng()) / float(UINT32_MAX);  // Random [0, 1]
```

---

## Usage Examples

### Sample Gaussians by Gradient

```cpp
// Compute weights from gradient magnitudes
std::vector<float> weights(n_gaussians);
for (int i = 0; i < n_gaussians; ++i) {
    weights[i] = gradient_magnitudes[i];
}

// Sample 100 Gaussians to duplicate
auto selected = multinomial_cpu_with_replacement(
    weights.data(),
    weights.size(),
    100,
    seed
);

// Duplicate selected Gaussians
for (int idx : selected.to_cpu()) {
    duplicate_gaussian(idx);
}
```

### CUDA-based Sampling

```cpp
// Weights on GPU
GPUBuffer<float> weights = compute_weights_on_gpu();

// Sample on GPU
GPUBuffer<int> selected = multinomial_cuda_with_replacement(
    weights.data(),
    weights.size(),
    num_samples,
    seed,
    stream
);

// Use selected indices on GPU
apply_selection<<<blocks, threads, 0, stream>>>(
    selected.data(),
    selected.size()
);
```

---

## Integration with Strategy

```cpp
class MCMCStrategy : public StrategyBase {
    void step_impl(const RasterizeContext& ctx) override {
        // Compute importance weights
        auto weights = compute_importance_weights();
        
        // Sample Gaussians for modification
        auto sample_indices = multinomial_cuda_cpu_without_replacement(
            weights.data(),
            m_gaussians->size(),
            num_to_modify,
            m_params.seed + this_step(),
            ctx.stream
        );
        
        // Apply modifications
        for (int idx : sample_indices.to_cpu()) {
            split_or_clone_gaussian(idx);
        }
    }
};
```

---

## Random Seed Management

```cpp
// Deterministic training
uint64_t seed = config["seed"].get<uint64_t>();

// Different seed per step for variety
int step_seed = base_seed + current_step;

// Reproducible results
auto samples = multinomial_cpu_with_replacement(
    weights.data(),
    K,
    num_samples,
    seed  // Fixed seed for reproducibility
);
```

---

## Performance Notes

- **CPU implementations**: Good for small-scale sampling (< 10K samples)
- **CUDA implementations**: Better for large-scale sampling
- **Without replacement**: O(K log K) for sorting, use with care for large K
- **With replacement**: O(num_samples log K) per batch

---

## Weight Normalization

Weights are automatically normalized internally:

```cpp
// These are equivalent:
weights = {1.0, 2.0, 3.0};  // Sum = 6.0
weights = {0.1, 0.2, 0.3};  // Sum = 0.6 (same ratios)

// Both give same sampling distribution
```

Zero or negative weights are handled:
- Zero weights: Never selected
- Negative weights: Treated as zero (clamped)