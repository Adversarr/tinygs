# Core Module

The core module provides fundamental data structures for 3D Gaussian Splatting.

## Files

| File | Description |
|------|-------------|
| `gaussian.hpp` | Host-side Gaussian data structure |
| `gpu_gaussian.hpp` | GPU-side Gaussian data structure (SoA) |
| `camera.hpp` | Camera intrinsics model |
| `camera_ext.hpp` | Camera extrinsics |
| `camera_loader.hpp` | Camera loading utilities |
| `image.hpp` | Image data structure |
| `pointcloud.hpp` | Point cloud for initialization |

---

## Gaussian3d (Host)

Host-side representation of 3D Gaussians using Structure of Arrays:

```cpp
struct Gaussian3d {
    std::vector<vec3> means;              // 3D positions
    std::vector<float> opacities;         // Opacity [0,1]
    std::vector<vec4> rotations;          // Quaternion (w,x,y,z)
    std::vector<vec3> scales;             // Scale (sx, sy, sz)
    std::vector<vec3> sh_coefficient_0;   // DC color (RGB)
    std::vector<vec3> sh_coefficients_rest; // SH 1-15 (15*3 values)
};
```

---

## GPUGaussian3d (Device)

GPU-side Gaussian storage using Thrust device vectors:

```cpp
class GPUGaussian3d {
public:
    void copy_from_host(const Gaussian3d& gaussians);
    void copy_to_host(Gaussian3d& gaussians);
    
    size_t size() const;
    
    thrust::device_vector<vec3>& means();
    thrust::device_vector<float>& opacities();
    thrust::device_vector<vec4>& rotations();
    thrust::device_vector<vec3>& scales();
    thrust::device_vector<vec3>& sh_coefficient_0();
    thrust::device_vector<vec3>& sh_coefficients_rest();
    
    std::unique_ptr<GPUGaussian3d> clone();
    void memset(char value);
    void remove(char* kept_flag, int num_kept);
    void reorder(uint* indices, cudaStream_t stream = 0);
    void append(int num_dup);
    
    float scene_scale() const;
    void set_scene_scale(float scale);
    int get_sh_degree() const;
    void set_sh_degree(int degree);
};
```

### Key Methods

| Method | Description |
|--------|-------------|
| `copy_from_host` | Copy data from CPU to GPU |
| `copy_to_host` | Copy data from GPU to CPU |
| `clone` | Deep copy on GPU |
| `remove` | Remove Gaussians based on keep flags |
| `reorder` | Reorder Gaussians by indices |
| `append` | Add new Gaussians |

---

## Camera Intrinsics

```cpp
enum class CameraModel {
    Pinhole,
    Fisheye
};

struct CameraIntrinsics {
    int id;
    CameraModel model;
    int width, height;
    float fx, fy;      // Focal length
    float cx, cy;      // Principal point
    
    mat3x3 to_mat3() const;
};
```

---

## Camera Extrinsics

```cpp
struct CameraExtrinsics {
    quat rotation;     // Orientation quaternion
    vec3 translation;  // Position
    uint64_t cam_id;
    uint64_t frame_id;
    
    mat4x4 get_w2c() const;  // World-to-camera matrix
    mat4x4 get_c2w() const;  // Camera-to-world matrix
};
```

---

## Image

```cpp
struct ImageShape {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
};

enum class DataType {
    Float32,
    Float16,
    UInt8
};

struct Image {
    ImageShape shape;
    DataType dtype;
    void* data;
    
    size_t size() const;
    size_t bytes() const;
};
```

### Memory Layout

Images use CHW (Channel-Height-Width) format with tiled storage for cache efficiency:

```
Tile size: 8x8 pixels
Linear index = (tile_y * tiled_width + tile_x) * 64 + intra_tile_offset
```

---

## Point Cloud

```cpp
struct PointCloud {
    std::vector<vec3> points;
    std::vector<vec3> colors;  // Optional RGB colors
};

PointCloud load_point_cloud(const std::string& path);
```

Supports PLY and TXT formats.

---

## SingleCameraLoader

Manages camera intrinsics and extrinsics for a dataset:

```cpp
class SingleCameraLoader {
public:
    void load_intrinsics(const std::string& path);
    void load_extrinsics(const std::string& path);
    
    CameraIntrinsics get_intrinsics(uint64_t cam_id) const;
    CameraExtrinsics get_extrinsics(uint64_t timestamp) const;
    
    void interpolate_poses(int num_frames);
    void undistort_images();
};
```