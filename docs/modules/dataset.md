# Dataset Module

The dataset module handles loading images and camera parameters from various sources.

## Files

| File | Description |
|------|-------------|
| `dataset.hpp` | Base class and data structures |
| `png_folder.hpp` | Image folder dataset (PNG/JPG) |
| `video.hpp` | Video file dataset (MP4) |

---

## Dataset Types

| Type | Description | Use Case |
|------|-------------|----------|
| `png_folder` | Image folder | Standard COLMAP datasets |
| `video` | Video file | Recorded video datasets |

---

## Data Structure

```cpp
struct Data {
    Image image;         // RGB image (float32, pinned host memory)
    mat4x4 w2c;         // World-to-camera matrix
    mat3x3 K;           // Intrinsic matrix
    uuid_t cam_uid;     // Camera ID (0-based)
    uuid_t frame_idx;   // Frame ID (1-based)
    uuid_t timestamp;   // Unique timestamp identifier
};
```

---

## DatasetBase Interface

```cpp
class DatasetBase {
public:
    /// @brief Load dataset from disk
    virtual void load() = 0;
    
    /// @brief Configuration
    virtual void set_params(const json& params);
    virtual json get_params() const;
    
    /// @brief Number of frames
    virtual size_t size() const noexcept = 0;
    
    /// @brief Image dimensions
    virtual ImageShape image_shape() const = 0;
    
    /// @brief Access frame by index
    virtual Data operator[](size_t idx) const = 0;
    
    /// @brief Camera loader access
    SingleCameraLoader& get_camera_loader();
};
```

---

## PNG Folder Dataset

Loads images from a folder with separate camera files.

### Configuration

```json
{
    "type": "png_folder",
    "folder_path": "outputs/scene/images/",
    "extrinsics_file_path": "outputs/scene/extri.txt",
    "intrinsics_file_path": "outputs/scene/intri.txt",
    "interpolate": true,
    "undistortion": false,
    "extension": "png"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `folder_path` | string | Path to image folder |
| `extrinsics_file_path` | string | Path to extrinsics file |
| `intrinsics_file_path` | string | Path to intrinsics file |
| `interpolate` | bool | Interpolate camera poses |
| `undistortion` | bool | Apply undistortion |
| `extension` | string | Image extension (`png`, `jpg`) |

### Extrinsics File Format

Text file with camera extrinsics:

```
# extri.txt format
# timestamp cam_id qw qx qy qz tx ty tz
1 0 0.9999 0.001 0.002 0.003 -0.5 0.1 0.2
2 0 0.9998 0.002 0.001 0.004 -0.49 0.11 0.21
...
```

### Intrinsics File Format

Text file with camera intrinsics:

```
# intri.txt format
# camera_id model width height fx fy cx cy [distortion_params]
0 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0
1 PINHOLE 1920 1080 1000.5 1000.5 960.5 540.5
...
```

---

## Video Dataset

Loads frames from a video file.

### Configuration

```json
{
    "type": "video",
    "video_file_path": "path/to/video.mp4",
    "video_info_path": "path/to/videoInfo.txt"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_file_path` | string | Path to video file |
| `video_info_path` | string | Path to camera info file |

### Supported Formats

- MP4
- AVI
- MOV
- Other OpenCV-supported formats

---

## Creating a Dataset

```cpp
// Via factory
auto dataset = create_dataset("png_folder");

// Configure
dataset->set_params({
    {"folder_path", "outputs/scene/images/"},
    {"extrinsics_file_path", "outputs/scene/extri.txt"},
    {"intrinsics_file_path", "outputs/scene/intri.txt"},
    {"extension", "png"}
});

// Load from disk
dataset->load();

// Access data
size_t num_frames = dataset->size();
Data frame0 = (*dataset)[0];
Image image = frame0.image;
mat4x4 w2c = frame0.w2c;
mat3x3 K = frame0.K;
```

---

## Camera Model

```cpp
enum class CameraModel {
    Pinhole,   // Standard pinhole camera
    Fisheye    // Fisheye camera (with distortion)
};

struct CameraIntrinsics {
    int id;
    CameraModel model;
    int width, height;
    float fx, fy;   // Focal length
    float cx, cy;   // Principal point
};
```

---

## Camera Extrinsics

```cpp
struct CameraExtrinsics {
    quat rotation;      // Orientation quaternion (w, x, y, z)
    vec3 translation;   // Position in world coordinates
    uint64_t cam_id;
    uint64_t frame_id;
    
    mat4x4 get_w2c() const;  // World-to-camera matrix
    mat4x4 get_c2w() const;  // Camera-to-world matrix
};
```

---

## SingleCameraLoader

Manages camera intrinsics and extrinsics:

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

---

## Usage Example

```cpp
// Create and configure dataset
auto train_dataset = create_dataset("png_folder");
train_dataset->set_params({
    {"folder_path", "data/train/images/"},
    {"extrinsics_file_path", "data/train/extri.txt"},
    {"intrinsics_file_path", "data/train/intri.txt"}
});
train_dataset->load();

// Create test dataset (optional)
auto test_dataset = create_dataset("png_folder");
test_dataset->set_params({
    {"folder_path", "data/test/images/"},
    {"extrinsics_file_path", "data/test/extri.txt"},
    {"intrinsics_file_path", "data/test/intri.txt"}
});
test_dataset->load();

// Create dataloaders
auto train_loader = create_dataloader("async", train_dataset);
auto test_loader = create_dataloader("simple", test_dataset);
```

---

## Pose Interpolation

When `interpolate: true`, camera poses are interpolated between keyframes:

```cpp
// Useful for video with sparse camera tracking
// Interpolates between tracked keyframes
```

---

## Undistortion

When `undistortion: true`, images are undistorted using camera distortion parameters:

```cpp
// Requires distortion coefficients in intrinsics file
// Useful for fisheye or distorted cameras
```