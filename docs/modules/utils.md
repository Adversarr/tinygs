# Utils Module

The utils module provides utility functions and helper classes.

## Files

| File | Description |
|------|-------------|
| `file.hpp` | File I/O and path utilities |
| `scope_timer.hpp` | RAII timing utilities |
| `image_format.hpp` | GPU image format conversion |
| `inspect_change.hpp` | Change detection utilities |
| `stbi_wrapper.cpp` | STB image loading wrapper |

---

## File Utilities

### Functions

```cpp
namespace tinygs {

/// @brief Read all non-empty lines from a text file
/// @param path Path to file
/// @return Vector of lines
std::vector<std::string> readlines(const std::string& path);

/// @brief List all files in a directory
/// @param path Directory path
/// @param relative Return relative paths (default: false)
/// @return Vector of file paths
std::vector<std::string> list_folder(const std::string& path, bool relative = false);

/// @brief Ensure a directory exists (create if needed)
/// @param path Directory path
void ensure(const std::string& path);

}
```

### Usage

```cpp
// Read lines from file
auto lines = readlines("config.txt");
for (const auto& line : lines) {
    // Process line
}

// List files in directory
auto files = list_folder("images/", true);
for (const auto& file : files) {
    log_info("Found: {}", file);
}

// Create directory if needed
ensure("output/checkpoints/");
```

---

## Scope Timer

RAII-based timing utility for profiling.

### GlobalTimerRegistry

```cpp
class GlobalTimerRegistry {
public:
    struct TimerStats {
        size_t count;
        double total_time;
        double min_time;
        double max_time;
        double average_time() const;
    };
    
    static GlobalTimerRegistry& get_instance();
    
    void record_time(const std::string& name, double time_ms);
    const TimerStats* get_stats(const std::string& name) const;
    void print_all_stats() const;
    void clear();
};
```

### ScopeTimer

```cpp
class ScopeTimer {
public:
    explicit ScopeTimer(const std::string& name);
    ~ScopeTimer();  // Automatically records time
    
    // Non-copyable, non-movable
};
```

### Usage

```cpp
// Manual usage
{
    ScopeTimer timer("render_pass");
    rasterizer->forward(ctx);
}  // Time recorded here

// Macro usage
void train_step() {
    TINYGS_TIMER_THIS_FUNCTION();  // Time entire function
    
    {
        TINYGS_TIMER("forward");
        rasterizer->forward(ctx);
    }
    
    {
        TINYGS_TIMER("backward");
        rasterizer->backward(ctx);
    }
}

// Print statistics
GlobalTimerRegistry::get_instance().print_all_stats();
```

### Output Example

```
Timer Statistics:
  forward: count=1000, avg=12.3ms, min=10.1ms, max=25.6ms, total=12300ms
  backward: count=1000, avg=8.5ms, min=7.2ms, max=15.3ms, total=8500ms
```

---

## Image Format Utilities

GPU image format conversion.

### Functions

```cpp
namespace tinygs {

/// @brief Convert image data type
void convert_image_dtype(
    const Image& src,
    Image& dst,
    cudaStream_t stream = 0
);

/// @brief Convert RGB to BGR (or vice versa)
void swap_channels(
    Image& image,
    cudaStream_t stream = 0
);

/// @brief Convert to RGBA
void add_alpha_channel(
    const Image& src,
    Image& dst,
    float alpha = 1.0f,
    cudaStream_t stream = 0
);

}
```

---

## Inspect Change

Detect changes between frames for adaptive training.

### Functions

```cpp
namespace tinygs {

/// @brief Compute difference between images
float compute_image_diff(
    const Image& img1,
    const Image& img2,
    cudaStream_t stream = 0
);

/// @brief Detect significant changes
bool has_significant_change(
    const Image& current,
    const Image& previous,
    float threshold = 0.1f,
    cudaStream_t stream = 0
);

}
```

---

## STB Image Wrapper

Wrapper for STB image loading library.

### Functions

```cpp
namespace tinygs {

/// @brief Load image from file
/// @param path Image file path
/// @return Image data (CPU memory)
Image load_image(const std::string& path);

/// @brief Save image to file
/// @param path Output path
/// @param image Image data
void save_image(const std::string& path, const Image& image);

}
```

### Supported Formats

- PNG
- JPEG
- BMP
- TGA
- PSD
- GIF
- HDR
- PIC

---

## Common Constants

```cpp
// Tile sizes
constexpr uint32_t kImageTile = 8;  // 8x8 pixel tiles

// Spherical harmonics
constexpr int kMaxSphericalHarmonicsDegree = 3;
constexpr int kMaxSphericalHarmonicsCoefficients = 16;

// CUDA
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t N_THREADS_LINEAR = 128;
```

---

## Utility Functions

```cpp
// Math utilities
template <typename T>
TINYGS_HOST_DEVICE T div_round_up(T val, T divisor);

template <typename T>
TINYGS_HOST_DEVICE T next_multiple(T val, T divisor);

TINYGS_HOST_DEVICE bool is_pot(uint32_t val);
TINYGS_HOST_DEVICE uint32_t next_pot(uint32_t v);

// Image indexing
TINYGS_HOST_DEVICE uint32_t get_linear_index(uint32_t i, uint32_t j, uint32_t width);
TINYGS_HOST_DEVICE uint32_t get_tile_index(uint32_t i, uint32_t j, uint32_t width);

// Constants
TINYGS_HOST_DEVICE float PI();
```

---

## Usage Examples

### File Operations

```cpp
// Read configuration
auto config_lines = readlines("config.txt");

// List dataset files
auto image_files = list_folder("data/images/");
std::sort(image_files.begin(), image_files.end());

// Create output directory
ensure("output/checkpoints/");
```

### Timing

```cpp
// Profile training steps
void train() {
    GlobalTimerRegistry::get_instance().clear();
    
    for (int step = 0; step < max_steps; ++step) {
        TINYGS_TIMER("train_step");
        train_step();
    }
    
    GlobalTimerRegistry::get_instance().print_all_stats();
}
```

### Image Loading

```cpp
// Load image from disk
Image image = load_image("texture.png");

// Process on GPU
// ...

// Save result
save_image("output/result.png", result_image);
```