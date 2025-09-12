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
from depth import DepthEstimator


PC_FILE = "/data/accgs/1747834320424/inputs/slam/points3D.txt"
EXTRIN_FILE = "/data/accgs/1747834320424/inputs/slam/images.txt"
INTRIN_FILE = "/data/accgs/1747834320424/inputs/slam/cameras.txt"
DEPTH_FILE = "./depth_output.png"
OUT_PIX_FILE = "pixels_cam0.txt"
OUT_PLOT = "pixels_cam0.png"
INPUT_FOLDER = "/data/accgs/1747834320424/inputs/images_480x640_1"

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

point_cloud = PointCloudManipulator(points, colors)
point_cloud.update(
    PointCloudUpdateConfig(
        enable_statistical_outlier_removal=True,
        enable_voxel_downsampling=False,
    )
)
points = np.asarray(point_cloud.pcd.points)
colors = np.asarray(point_cloud.pcd.colors)
point_ids = np.arange(points.shape[0])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud.pcd)

estimated = None

depth_anything = DepthEstimator()

valid_scales = []
valid_scores = []
histories = []

# First round.

for cam in camera_extrinsics:
    cam_id = cam.img_id
    png_file = f"{INPUT_FOLDER}/{cam_id:04d}.png"
    print(f"Processing camera {cam_id}, image file: {png_file}")
    rgb = np.array(Image.open(png_file))
    rel_depth: np.ndarray = depth_anything.predict_depth(rgb) # type: ignore
    start = time()
    u_in, v_in, z_in, pids_in, mask_zpos = project_points_to_camera(points, point_ids, camera_intrinsics, calc_w2c(cam))
    aligner = DepthAlignment(max_trials=1000, random_state=42)
    result: DepthAlignmentResult = aligner.estimate_scale(points, point_ids, camera_intrinsics, cam, rel_depth)
    print(f"Estimated scale: {result.scale}, offset: {result.offset}, inliers: {np.sum(result.inlier_mask)} / {len(result.inlier_mask)} score: {result.score}")

    if result.scale < 0:
        print("Warning: estimated scale is negative, skipping this frame.")
        continue

    # unproject depth map to 3D points
    points_world_unproj = aligner.unproject_depth_map(
        rel_depth,
        camera_intrinsics,
        cam,
        downsampling_factor=4,
        scale=result.scale,
        offset=result.offset,
    )
    end = time()
    print(f"Time for alignment: {end - start:.4f} seconds")

    valid_scales.append(result.scale)
    valid_scores.append(result.score)

    if estimated is None:
        estimated = o3d.geometry.PointCloud()
        estimated.points = o3d.utility.Vector3dVector(points_world_unproj)
        estimated.paint_uniform_color([0.3, 0.3, 0])
        vis.add_geometry(estimated)
    else:
        estimated.points = o3d.utility.Vector3dVector(points_world_unproj)
        vis.update_geometry(estimated)

    vis.poll_events()
    vis.update_renderer()


plt.figure(figsize=(8, 6))
plt.hist(valid_scales, bins=40, color='blue', alpha=0.7)
plt.xlabel('Estimated Scale')
plt.ylabel('Frequency')
plt.title('Histogram of Estimated Scales from Depth Alignment')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(valid_scores, bins=40, color='green', alpha=0.7)
plt.xlabel('RANSAC Score')
plt.ylabel('Frequency')
plt.title('Histogram of RANSAC Scores from Depth Alignment')
plt.show()