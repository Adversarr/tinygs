# Initialization Module

The initialization module handles creating initial Gaussian primitives from point clouds.

## Files

| File | Description |
|------|-------------|
| `initialization.hpp` | Base class |
| `knn.hpp` | KNN-based initialization |
| `random.hpp` | Random initialization |

---

## Initialization Types

| Type | Description | Use Case |
|------|-------------|----------|
| `knn` | K-Nearest Neighbors | Default, best quality |
| `random` | Random sampling | Simple baseline |

---

## InitializationBase Interface

```cpp
class InitializationBase {
public:
    virtual ~InitializationBase() = default;
    
    /// @brief Initialize Gaussians from point cloud
    virtual void initialize(const PointCloud& pointcloud) = 0;
    
    /// @brief Get initialized Gaussians
    const Gaussian3d& gaussians() const;
    
    /// @brief Configuration
    virtual void set_params(const json& params) = 0;
    virtual json get_params() const = 0;
};
```

---

## KNN Initialization

Initializes Gaussians using K-Nearest Neighbors for scale estimation:

### Configuration

```json
{
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
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_neighbors` | int | 8 | KNN neighbors for scale estimation |
| `default_distance` | float | 0.01 | Default scale if KNN fails |
| `init_opacity` | float | 0.1 | Initial Gaussian opacity |
| `init_scaling` | float | 0.6 | Scale multiplier |
| `sh_degree` | int | 3 | Maximum SH degree |
| `enable_radius_outlier_removal` | bool | false | Remove outlier points |
| `radius` | float | 0.05 | Outlier removal radius |
| `nb_points` | int | 16 | Min points in radius for outliers |
| `min_distance` | float | 1e-7 | Minimum scale distance |

### Algorithm

1. **Load Point Cloud**: Read points and colors from PLY/TXT
2. **Optional Outlier Removal**: Remove sparse points
3. **KNN Scale Estimation**: For each point:
   - Find K nearest neighbors
   - Compute average distance to neighbors
   - Use as initial scale
4. **Initialize Gaussians**:
   - Position: point position
   - Scale: KNN distance * init_scaling
   - Rotation: identity quaternion
   - Opacity: init_opacity
   - Color: SH from point color

```cpp
class KNNInitialization : public InitializationBase {
    void initialize(const PointCloud& pointcloud) override;
    
    // Parameters
    int m_num_neighbors = 8;
    float m_default_distance = 0.01f;
    float m_init_opacity = 0.1f;
    float m_init_scaling = 0.6f;
    int m_sh_degree = 3;
};
```

---

## Random Initialization

Initializes Gaussians with random positions and scales:

### Configuration

```json
{
    "type": "random",
    "num_gaussians": 10000,
    "init_opacity": 0.1,
    "init_scale": 0.01
}
```

---

## Creating an Initializer

```cpp
// Via factory
auto initializer = create_initialization("knn");

// Configure
initializer->set_params({
    {"num_neighbors", 8},
    {"init_opacity", 0.1},
    {"init_scaling", 0.6}
});

// Initialize from point cloud
PointCloud pc = load_point_cloud("scene/points.ply");
initializer->initialize(pc);

// Get initialized Gaussians
const Gaussian3d& gaussians = initializer->gaussians();
```

---

## Point Cloud Format

Supported formats:

### PLY Format

```
ply
format ascii 1.0
element vertex 10000
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0.0 0.0 0.0 128 128 128
...
```

### TXT Format

```
# x y z r g b
0.0 0.0 0.0 0.5 0.5 0.5
0.1 0.2 0.3 0.6 0.4 0.2
...
```

---

## PointCloud Structure

```cpp
struct PointCloud {
    std::vector<vec3> points;   // 3D positions
    std::vector<vec3> colors;   // RGB colors [0,1]
};

// Loading function
PointCloud load_point_cloud(const std::string& path);
```

---

## Scale Estimation Detail

KNN-based scale estimation:

```cpp
// For each point p_i:
// 1. Find K nearest neighbors
// 2. Compute average distance
float avg_distance = 0;
for (int j = 0; j < K; ++j) {
    avg_distance += distance(p_i, neighbor_j);
}
avg_distance /= K;

// 3. Set scale
scale_i = avg_distance * init_scaling;
```

This produces scales proportional to local point density.

---

## Spherical Harmonics

Colors are converted to SH coefficients:

```cpp
// DC coefficient (degree 0)
sh_coefficient_0 = color;

// Higher degrees (degree 1-3) initialized to zero
sh_coefficients_rest = zeros;
```

During training, SH degrees progressively activate:
- Step 0: Degree 0 (DC only)
- Step 1000: Degree 1
- Step 2000: Degree 2
- Step 3000: Degree 3

---

## Usage Example

```cpp
// Load point cloud
PointCloud pc = load_point_cloud("outputs/scene/points.ply");
log_info("Loaded {} points", pc.points.size());

// Create initializer
auto initializer = create_initialization("knn");
initializer->set_params({
    {"num_neighbors", 8},
    {"init_opacity", 0.1},
    {"init_scaling", 0.6}
});

// Initialize
initializer->initialize(pc);
const Gaussian3d& gaussians = initializer->gaussians();
log_info("Initialized {} Gaussians", gaussians.size());

// Copy to GPU
auto gpu_gaussians = std::make_shared<GPUGaussian3d>();
gpu_gaussians->copy_from_host(gaussians);
```

---

## Outlier Removal

When enabled, removes points with few neighbors:

```cpp
// Remove points with fewer than nb_points within radius
if (enable_radius_outlier_removal) {
    // For each point:
    // Count neighbors within radius
    // If count < nb_points, remove point
}
```

Useful for cleaning noisy point clouds.