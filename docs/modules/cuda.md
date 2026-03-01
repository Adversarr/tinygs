# CUDA Module

The CUDA module provides GPU utilities, memory management, and parallel primitives.

## Files

| File | Description |
|------|-------------|
| `common_host.hpp` | Host-side CUDA helpers, logging, device queries |
| `common_device.cuh` | Device-side CUDA utilities |
| `gpu_memory.hpp` | GPU memory allocation and management |
| `multi_stream.hpp` | Multi-stream execution utilities |
| `reduce.hpp` | Parallel reduction kernels |
| `stat.hpp` | GPU statistics computation |
| `vec.hpp` | Vector types and operations |
| `cuda_graph.hpp` | CUDA graph utilities |

---

## Memory Management

### GPUMemory

Simple GPU memory wrapper:

```cpp
template <typename T>
class GPUMemory {
public:
    GPUMemory() = default;
    explicit GPUMemory(size_t size);
    
    void allocate(size_t size);
    void free();
    
    T* data();
    const T* data() const;
    size_t size() const;
    
    void copy_from_host(const std::vector<T>& host_data);
    void copy_to_host(std::vector<T>& host_data) const;
    
    void memset(int value, cudaStream_t stream = 0);
};
```

### GPUMemoryArena

Pooled memory allocator for intermediate buffers:

```cpp
class GPUMemoryArena {
public:
    void* allocate(size_t size);
    void reset();
    
    template <typename T>
    T* allocate(size_t count);
};
```

### GPUBuffer

Typed buffer with automatic management:

```cpp
template <typename T>
class GPUBuffer {
public:
    GPUBuffer() = default;
    explicit GPUBuffer(size_t size);
    
    T* data();
    size_t size() const;
    
    std::vector<T> to_cpu() const;
    void from_cpu(const std::vector<T>& data);
};
```

---

## Device Queries

```cpp
int cuda_device();                          // Current device ID
int cuda_device_count();                    // Number of devices
std::string cuda_device_name(int device);   // Device name
uint32_t cuda_compute_capability(int dev);  // Compute capability
size_t cuda_max_shmem(int device);          // Max shared memory
uint32_t cuda_max_registers(int device);    // Max registers
MemoryInfo cuda_memory_info();              // Memory usage
```

---

## Logging

```cpp
#define log_info(...) SPDLOG_INFO(__VA_ARGS__)
#define log_debug(...) SPDLOG_DEBUG(__VA_ARGS__)
#define log_warning(...) SPDLOG_WARN(__VA_ARGS__)
#define log_error(...) SPDLOG_ERROR(__VA_ARGS__)
```

---

## Error Checking

```cpp
#define CUDA_CHECK_THROW(x)  // Throw on error
#define CUDA_CHECK_PRINT(x)   // Print on error
#define CU_CHECK_THROW(x)     // Driver API throw
#define CU_CHECK_PRINT(x)     // Driver API print
```

---

## Parallel Primitives

### Linear Kernel Launch

```cpp
template <typename K, typename T, typename... Args>
void linear_kernel(K kernel, uint32_t shmem, cudaStream_t stream, T n, Args... args);
```

### Parallel For

```cpp
template <typename F>
void parallel_for_gpu(cudaStream_t stream, size_t n, F&& fun);

template <typename F>
void parallel_for_gpu_aos(cudaStream_t stream, size_t n, uint32_t dims, F&& fun);

template <typename F>
void parallel_for_gpu_soa(cudaStream_t stream, size_t n, uint32_t dims, F&& fun);
```

---

## Reduction

```cpp
void reduce_sum(const float* input, float* output, size_t n, cudaStream_t stream);
void reduce_max(const float* input, float* output, size_t n, cudaStream_t stream);
void reduce_min(const float* input, float* output, size_t n, cudaStream_t stream);
```

---

## Statistics

```cpp
void compute_mean_variance(const float* data, float* mean, float* var, 
                           size_t n, cudaStream_t stream);
```

---

## Vector Types

```cpp
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat3 = glm::mat3;
using mat4 = glm::mat4;
using mat3x3 = glm::mat3;
using mat4x4 = glm::mat4;
using quat = glm::quat;
```

---

## Constants

```cpp
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t N_THREADS_LINEAR = 128;
constexpr uint32_t BATCH_SIZE_GRANULARITY = 256;
constexpr uint32_t kImageTile = 8;
constexpr int kMaxSphericalHarmonicsDegree = 3;
constexpr int kMaxSphericalHarmonicsCoefficients = 16;
```

---

## Utility Functions

```cpp
TINYGS_HOST_DEVICE float PI();
TINYGS_HOST_DEVICE T div_round_up(T val, T divisor);
TINYGS_HOST_DEVICE T next_multiple(T val, T divisor);
TINYGS_HOST_DEVICE bool is_pot(T val);
TINYGS_HOST_DEVICE uint32_t next_pot(uint32_t v);

TINYGS_HOST_DEVICE uint32_t get_linear_index(uint32_t i, uint32_t j, uint32_t width);
TINYGS_HOST_DEVICE uint32_t get_tile_index(uint32_t i, uint32_t j, uint32_t width);
```