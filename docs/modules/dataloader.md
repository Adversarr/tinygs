# DataLoader Module

The dataloader module provides efficient data loading and batching for training.

## Files

| File | Description |
|------|-------------|
| `dataloader.hpp` | Base class and data structures |
| `simple.hpp` | Synchronous single-stream loader |
| `async.hpp` | Asynchronous multi-stream loader |

---

## DataLoader Types

| Type | Description | Use Case |
|------|-------------|----------|
| `simple` | Synchronous loading | Debugging, simple use cases |
| `async` | Asynchronous with prefetch | Production training (recommended) |

---

## GPUBatchInput

Camera and rendering parameters:

```cpp
struct GPUBatchInput {
    uint32_t width, height;  // Image dimensions
    float near, far;         // Near/far clipping planes
    mat3x3 K;               // Intrinsic matrix (3x3)
    mat4x4 w2c;             // World-to-camera matrix (4x4)
    uuid_t timestamp;       // Frame/camera identifier
};
```

---

## GPUBatchOutput

Rendering output:

```cpp
struct GPUBatchOutput {
    Image image;  // RGB image in CHW format
};
```

---

## GPUBatchInputOutput

Combined input/output structure:

```cpp
struct GPUBatchInputOutput {
    GPUBatchInput input;
    GPUBatchOutput output;
};
```

---

## DataLoaderParams

```cpp
struct DataLoaderParams {
    DataType data_type = DataType::Float32;  // Float32 or Float16
};
```

---

## DataLoaderBase Interface

```cpp
class DataLoaderBase {
public:
    explicit DataLoaderBase(std::shared_ptr<DatasetBase> dataset);
    
    /// @brief Get next batch
    virtual GPUBatchInputOutput next() = 0;
    
    /// @brief Reset to initial state
    virtual void reset();
    
    /// @brief Set output image shape
    virtual void set_output_shape(const ImageShape& shape);
    
    /// @brief Transfer data to GPU
    void transfer_gpu(cudaStream_t stream, const Image& gpu_data, const Image& host_data);
    
    /// @brief Access underlying dataset
    std::shared_ptr<DatasetBase> get_dataset() const;
    
    /// @brief Configuration
    virtual void set_params(const json& params);
    virtual json get_params() const;
};
```

---

## Simple DataLoader

Synchronous single-stream loader:

```cpp
class SimpleDataLoader : public DataLoaderBase {
    // Loads data synchronously on each next() call
    // Simple but may cause GPU idle time
};
```

**Pros:**
- Simple implementation
- Easy to debug

**Cons:**
- GPU may wait for data loading
- Less efficient for training

---

## Async DataLoader

Asynchronous multi-queue loader with prefetching:

```cpp
class AsyncDataLoader : public DataLoaderBase {
    // Uses multiple queues
    // Prefetches next batch while GPU processes current
};
```

**Features:**
- Multiple queues for overlap
- Prefetches data ahead of time
- Hides data transfer latency

**Pros:**
- Better GPU utilization
- Faster training

---

## Creating a DataLoader

```cpp
// Create dataset first
auto dataset = create_dataset("image");
dataset->set_params({
    {"root_path", "outputs/scene/train/"},
    {"extension", "png"},
    {"resolution", -1},
    {"resolution_scale", 1.0f}
});
dataset->load();

// Create dataloader
auto dataloader = create_dataloader("async", dataset);

// Configure
json params = {{"data_type", "float16"}};
dataloader->set_params(params);

// Set output shape (optional)
dataloader->set_output_shape({width, height, 3});
```

---

## Usage Example

```cpp
// Setup
auto dataset = create_dataset("image");
dataset->set_params({
    {"root_path", "outputs/scene/train/"},
    {"extension", "png"},
    {"resolution", -1},
    {"resolution_scale", 1.0f}
});
dataset->load();

auto dataloader = create_dataloader("async", dataset);
dataloader->set_params({{"data_type", "float16"}});

// Training loop
for (int step = 0; step < max_steps; ++step) {
    // Get next batch
    auto batch = dataloader->next();
    
    // Use batch.input for camera parameters
    // Use batch.output.image for target image
    ctx.fwd_input = batch.input;
    
    // Forward pass...
}
```

---

## Image Format

Images use CHW (Channel-Height-Width) format:

```cpp
// Image dimensions
ImageShape shape = {width, height, channels};  // channels = 3 for RGB

// Data types
DataType dtype = DataType::Float32;  // or Float16, UInt8
```

---

## Memory Layout

Images use tiled storage for cache efficiency:

```cpp
constexpr uint32_t kImageTile = 8;  // 8x8 pixel tiles

// Linear index calculation
uint32_t idx = get_linear_index(i, j, width);
```

---

## Data Transfer

```cpp
// Transfer from host to device
Image host_image = dataset->get_image(idx);  // Host memory
Image gpu_image;  // Device memory

dataloader->transfer_gpu(stream, gpu_image, host_image);
```

Handles:
- Host-to-device copy
- Type conversion (UInt8 -> Float)
- Format validation