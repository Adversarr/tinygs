from time import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.cm as cm
from camera import (
    load_camera_intrinsics,
    load_camera_extrinsics,
    load_point_cloud,
    calc_c2w,
    calc_w2c,
    project_points_to_camera,
    CameraIntrinsic,
    CameraExtrinsic,
)
from depth_alignment import DepthAlignment, DepthAlignmentResult
from point_cloud import PointCloudManipulator, PointCloudUpdateConfig


PC_FILE = "/data/accgs/1747834320424/inputs/slam/points3D.txt"
EXTRIN_FILE = "/data/accgs/1747834320424/inputs/slam/images.txt"
INTRIN_FILE = "/data/accgs/1747834320424/inputs/slam/cameras.txt"
DEPTH_FILE = "./depth_output.png"
OUT_PIX_FILE = "pixels_cam0.txt"
OUT_PLOT = "pixels_cam0.png"

# --- Load camera intrinsics ---
camera_intrinsics = load_camera_intrinsics(INTRIN_FILE)
print(f"Camera: w={camera_intrinsics.width}, h={camera_intrinsics.height}, fx={camera_intrinsics.fx}, fy={camera_intrinsics.fy}, cx={camera_intrinsics.cx}, cy={camera_intrinsics.cy}")

# --- Load camera extrinsics ---
camera_extrinsics, qs, ts = load_camera_extrinsics(EXTRIN_FILE)
print(f"Parsed {len(qs)} camera poses")

# --- Load point cloud ---
point_ids, points, colors = load_point_cloud(PC_FILE)
print(f"Loaded {points.shape[0]} 3D points")

start_time = time()
# --- 选择第一个相机进行投影（如需别的相机，改变 idx） ---
cam_idx = 0

# Create CameraExtrinsic object for the selected camera
extrinsic = CameraExtrinsic(0, rotation=qs[cam_idx], translation=ts[cam_idx])

# Get transformation matrix
T_wc = calc_w2c(extrinsic)
T_cw = calc_c2w(extrinsic)

# Extract rotation matrix and translation vector
R_wc = T_wc[0:3, 0:3]  # Rotation matrix from world to camera
t_wc = T_wc[0:3, 3]    # Translation vector from world to camera

# Project points to camera and filter visible points
u_in, v_in, z_in, pids_in, mask_zpos = project_points_to_camera(points, point_ids, camera_intrinsics, T_wc)

# [0, 1], float32, shape (H, W)
rel_depth = np.array(Image.open(DEPTH_FILE)).astype(np.float32) / 255.0
depth_at_proj = rel_depth[v_in.astype(np.int32), u_in.astype(np.int32)]
# inverse z.
inv_z_in = 1.0 / (z_in + 1e-8)

# Initialize depth alignment with RANSAC
depth_alignment = DepthAlignment(max_trials=1000, random_state=42)
alignment_result = depth_alignment.estimate_scale(
    points, point_ids, camera_intrinsics, extrinsic, rel_depth
)
print("RANSAC coef:", alignment_result.scale, "intercept:", alignment_result.offset)

inlier = alignment_result.inlier_mask
end_time = time()

print(f"Algorithm done in {end_time - start_time:.2f} seconds")

# Plot x, y and the line
plt.figure(figsize=(8, 6))
plt.scatter(alignment_result.relative_depth_values, alignment_result.inverse_depth_values, s=1, c='k', alpha=0.5, label='Data points')
x_line = np.linspace(alignment_result.relative_depth_values.min(), alignment_result.relative_depth_values.max(), 100)
y_line = alignment_result.ransac_model.predict(x_line.reshape(-1, 1))
plt.plot(x_line, y_line, color='r', linewidth=2, label='RANSAC fit')
plt.xlabel('Depth from image (normalized)')
plt.ylabel('Inverse depth from 3D points (1/m)')
plt.title('Depth vs Inverse Depth with RANSAC Fit')
plt.legend()
plt.savefig("depth_vs_invz.png", dpi=300)
plt.close()

# Show the distribution
dist_data = alignment_result.relative_depth_values * z_in
plt.hist(dist_data, bins=100, color='blue', alpha=0.7)
plt.xlabel('Depth Difference (image depth * 3D point depth)')
plt.ylabel('Frequency')
plt.title('Distribution of Depth Differences')
plt.savefig("depth_difference_distribution.png", dpi=300)
plt.close()

# --- 可视化（以像素坐标绘图，v 向下） ---
plt.figure(figsize=(8, 8 * camera_intrinsics.height / camera_intrinsics.width))  # 保持纵横比
plt.scatter(u_in, v_in, s=0.5, c=dist_data, alpha=0.8)
plt.xlim(0, camera_intrinsics.width)
plt.ylim(camera_intrinsics.height, 0)  # 反转 y 轴使得像素 (0,0) 在左上
plt.xlabel('u (pixels)')
plt.ylabel('v (pixels)')
plt.title(f'Camera {cam_idx} projection: {u_in.size} points')
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300)
plt.close()
print(f"Saved visualization to {OUT_PLOT}")


# --- 可视化点云和相机位置（使用 open3d） ---
geom = []

# 点云
# Normalize dist_data to [0, 1] range for coloring
dist_normalized = (dist_data - dist_data.min()) / (dist_data.max() - dist_data.min())
# Create colors using a colormap (e.g., viridis)
colors = cm.viridis(dist_normalized)[:, :3]  # Take RGB, ignore alpha
colors[inlier == False] = [0.5, 0.5, 0.5]  # Outliers in gray

# Use PointCloudManipulator for the main point cloud
pcd_manipulator = PointCloudManipulator(points[mask_zpos], colors)
geom.append(pcd_manipulator.pcd)

# 相机位置
cam_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
cam_mesh.transform(T_cw)
geom.append(cam_mesh)

# Unproject the depth map to a point cloud for better visualization
points_world_unproj = depth_alignment.unproject_depth_map(
    rel_depth,
    camera_intrinsics,
    extrinsic,
    downsampling_factor=1,
    scale=alignment_result.scale,
    offset=alignment_result.offset
)

# Create point cloud for unprojected depth map
# Use PointCloudManipulator for the depth map point cloud
red_color = np.tile([1.0, 0.0, 0.0], (len(points_world_unproj), 1))
pcd_depth_manipulator = PointCloudManipulator(points_world_unproj, red_color)
pcd_depth = pcd_depth_manipulator.pcd
# Use PointCloudUpdateConfig for statistical outlier removal and voxel downsampling
update_config = PointCloudUpdateConfig(
    sor_nb_neighbors=20,
    sor_std_ratio=2.0,
    ds_voxel_size=0.01
)
pcd_depth_manipulator.update(update_config)
pcd_depth = pcd_depth_manipulator.pcd
print(f"Depth map point cloud has {len(pcd_depth.points)} points after downsampling")
geom.append(pcd_depth)


# Create world coordinate frame
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
geom.append(world_frame)

# 可视化
o3d.visualization.draw_geometries(geom)