import math
import os
from time import perf_counter, time
from typing import List
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from camera import (
    CameraExtrinsic,
    CameraIntrinsic,
    load_camera_intrinsics,
    load_camera_extrinsics,
    load_point_cloud,
    calc_w2c,
    project_points_to_camera,
)
from pathlib import Path
from depth_alignment import DepthAlignment, DepthAlignmentResult
from point_cloud import PointCloudManipulator, PointCloudUpdateConfig
from depth import DepthEstimator
import cv2
from tqdm import tqdm
from collections import OrderedDict
from argparse import ArgumentParser

import trimesh

class Aligner:
    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        images: List[str],
        camera_extrinsics: List[CameraExtrinsic],
        camera_intrinsics: CameraIntrinsic,
    ):
        self.depth_anything = DepthEstimator()
        self.init_point_cloud = PointCloudManipulator(points, colors)

        self.point_cloud_running = PointCloudManipulator(points, colors)
        self.histories = OrderedDict()  # cam_id -> (depth, rgb, cam)
        self.camera_extrinsics = camera_extrinsics
        self.camera_intrinsics = camera_intrinsics
        self.images = images
        K = np.array([
            [camera_intrinsics.fx, 0, camera_intrinsics.cx],
            [0, camera_intrinsics.fy, camera_intrinsics.cy],
            [0, 0, 1]
        ])
        distortion = np.array([
            camera_intrinsics.k1,
            camera_intrinsics.k2,
            camera_intrinsics.p1,
            camera_intrinsics.p2
        ])

        w, h = camera_intrinsics.width, camera_intrinsics.height
        newK, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (w, h), alpha=0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, distortion, None, newK, (w, h), cv2.CV_32FC1) # type: ignore
        self.undistorted = CameraIntrinsic(
            width=w,
            height=h,
            fx=newK[0, 0],
            fy=newK[1, 1],
            cx=newK[0, 2],
            cy=newK[1, 2],
            k1=distortion[0],
            k2=distortion[1],
            k3=self.camera_intrinsics.k3,
            p1=distortion[2],
            p2=distortion[3],
        )

        self.interval_removal = 1
        self.downsampling = 3

    def run_ransac(self, points, cam: CameraExtrinsic, depth):
        aligner = DepthAlignment(max_trials=1000, stop_score=0.999)
        result: DepthAlignmentResult = aligner.estimate_scale(
            points, self.undistorted, cam, depth
        )
        return result, aligner

    def estimate_depth(self, img_path):
        img = Image.open(img_path)
        img = cv2.resize(np.array(img), (self.undistorted.width, self.undistorted.height))
        img_np = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        depth_map: np.ndarray = self.depth_anything.predict_depth(img_np)  # type: ignore
        return depth_map, img_np

    def best_alignment(self, avail: List[int], known):
        max_score = -1
        best_result = None
        best_cam_id = -1
        best_aligner = None
        running = self.point_cloud_running.xyz
        init = self.init_point_cloud.xyz

        # Limit the running's size to init's size
        scaler = 1 + math.log(1 + known)
        max_size = int(init.shape[0] * scaler)
        if running.shape[0] > max_size:
            c = np.random.choice(running.shape[0], max_size, replace=False)
            points = np.vstack([init, running[c]])
        elif known > 0:
            points = np.vstack([init, running])
        else:
            points = init

        for cam_id in avail:
            depth, _, cam = self.histories[cam_id]
            result, aligner = self.run_ransac(points, cam, depth)
            if result.score > max_score and result.scale > 0:
                max_score = result.score
                best_result = result
                best_cam_id = cam_id
                best_aligner = aligner
        return best_cam_id, best_result, best_aligner

    def run_finalize(self):
        # cam_poses = np.stack([e.translation for e in self.camera_extrinsics], axis=0)
        # self.point_cloud_running.remove_hidden_points(cam_poses)
        self.point_cloud_running.update(PointCloudUpdateConfig(
            enable_radius_outlier_removal=True,
            ror_radius=0.1,
            enable_statistical_outlier_removal=True,
            sor_std_ratio=2,
            enable_voxel_downsampling=True,
        ))
        self.point_cloud_running.add_points(self.init_point_cloud.xyz, self.init_point_cloud.rgb)

    def run(self):
        for (cam, img_path) in tqdm(zip(self.camera_extrinsics, self.images), total=len(self.images)):
            cam_id = cam.timestamp
            depth, rgb = self.estimate_depth(img_path)
            self.histories[cam_id] = (depth, rgb, cam)

        avail_intervals = [list(self.histories.keys())]
        iteration = 0
        estim_scale = -1
        prev_scales = []
        while avail_intervals:
            candidates = []
            for interval in avail_intervals:
                best_cam_id, best_result, best_aligner = self.best_alignment(interval, len(prev_scales))
                if best_cam_id != -1 and best_result is not None:
                    if (
                        best_result.score >= 0.96
                        and 0.1 * estim_scale < best_result.scale < 10 * estim_scale
                    ) or iteration < 2:
                        print(f"{iteration}: selected cam {best_cam_id} with score {best_result.score:.4f} and scale {best_result.scale:.4f}")
                        candidates.append((best_cam_id, best_result, best_aligner))
                    else:
                        print(f"{iteration}: best cam {best_cam_id} has low score {best_result.score:.4f}, skipping interval (scale={best_result.scale:.4f}, estim_scale={estim_scale:.4f})")

            if not candidates:
                print("No more good candidates, stopping.")
                return

            print(f"Iter {iteration}: Adding {len(candidates)} cameras to point cloud")
            unprojected_xyz = []
            unprojected_rgb = []
            for (cam_id, best_result, best_aligner) in candidates:
                depth, rgb, cam = self.histories[cam_id]
                points_world_unproj, mask = best_aligner.unproject_depth_map(
                    depth,
                    self.undistorted,
                    cam,
                    downsampling_factor=self.downsampling,
                    scale=best_result.scale,
                    offset=best_result.offset,
                )
                unprojected_xyz.append(points_world_unproj)
                rgb_out = rgb[::self.downsampling, ::self.downsampling].reshape(-1, 3) / 255.0
                unprojected_rgb.append(rgb_out[mask])
                prev_scales.append(best_result.scale)

            self.point_cloud_running.add_points(
                np.vstack(unprojected_xyz),
                np.vstack(unprojected_rgb)
            )
            self.point_cloud_running.update(PointCloudUpdateConfig(
                enable_radius_outlier_removal=True,
                ror_radius=0.2,
                sor_std_ratio=2.5,
                enable_voxel_downsampling=False,
            ))

            # Update the available intervals
            used_cam_ids = {cam_id for (cam_id, _, _) in candidates}
            new_intervals = []
            for interval in avail_intervals:
                current_interval = []
                for ith, cam_id in enumerate(interval):
                    beg = max(0, ith - self.interval_removal)
                    end = min(len(interval) - 1, ith + self.interval_removal)
                    if any([interval[i] in used_cam_ids for i in range(beg, end + 1)]):
                        if current_interval:
                            new_intervals.append(current_interval)
                            current_interval = []
                    else:
                        current_interval.append(cam_id)
                if current_interval:
                    new_intervals.append(current_interval)
            avail_intervals = [interv for interv in new_intervals if len(interv) > self.interval_removal]
            print(avail_intervals)
            estim_scale = np.mean(prev_scales) if prev_scales else -1
            iteration += 1

    def export(self, filename):
        print(f"Exporting {self.point_cloud_running.xyz.shape[0]} points to {filename}")
        self.point_cloud_running.save_to_file(filename)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--working_dir", type=str, help="Working directory", default='out2/1747834320424')
    parser.add_argument("--downsampling", type=int, help="Downsampling factor", default=480)
    parser.add_argument('--t-interval', type=int, help="Time interval for alignment", default=4)
    args = parser.parse_args()
    PC_FILE = f"{args.working_dir}/points3D.ply"
    EXTRIN_FILE = f"{args.working_dir}/train_desired.txt"
    INTRIN_FILE = f"{args.working_dir}/intrinsics.txt"
    INPUT_FOLDER = f"{args.working_dir}/images/"
    OUT_FILE = f"{args.working_dir}/aligned_points.ply"
    TIME_FILE = f"{args.working_dir}/aligner_time.txt"

    # --- Load camera intrinsics ---
    camera_intrinsics = load_camera_intrinsics(INTRIN_FILE)
    print(f"Camera: w={camera_intrinsics.width}, h={camera_intrinsics.height}, fx={camera_intrinsics.fx}, fy={camera_intrinsics.fy}, cx={camera_intrinsics.cx}, cy={camera_intrinsics.cy}")

    downscale = camera_intrinsics.width / args.downsampling
    camera_intrinsics.width = int(np.round(camera_intrinsics.width / downscale))
    camera_intrinsics.height = int(np.round(camera_intrinsics.height / downscale))
    camera_intrinsics.fx /= downscale
    camera_intrinsics.fy /= downscale
    camera_intrinsics.cx /= downscale
    camera_intrinsics.cy /= downscale
    print(f"Downscaled camera: w={camera_intrinsics.width}, h={camera_intrinsics.height}")

    # --- Load camera extrinsics ---
    camera_extrinsics, qs, ts = load_camera_extrinsics(EXTRIN_FILE)
    print(f"Parsed {len(qs)} camera poses")
    camera_extrinsics = camera_extrinsics[::args.t_interval]
    qs = qs[::args.t_interval]
    ts = ts[::args.t_interval]

    # --- Load point cloud ---
    pc = trimesh.load(PC_FILE)
    points = np.array(pc.vertices)
    colors = np.array(pc.colors)[:, :3] / 255.0

    print(f"Loaded {points.shape[0]} 3D points")

    images = []
    for cam in camera_extrinsics:
        cam_id = cam.timestamp
        png_file = f"{INPUT_FOLDER}/{cam_id}.png"
        if os.path.exists(png_file):
            images.append(png_file)
        else:
            camera_extrinsics.remove(cam)

    print(f"Loaded {len(images)} images")
    if not images:
        print("Error: no images found")
        exit(1)

    start_time = perf_counter()
    try:
        aligner = Aligner(
            points,
            colors,
            images,
            camera_extrinsics,
            camera_intrinsics,
        )
        aligner.run()
        aligner.run_finalize()
        end_time = perf_counter()
        print(f"Alignment time: {end_time - start_time}")
        aligner.export(OUT_FILE)
    except Exception as e:
        print(f"Error: {e}")
        # use init point cloud
        aligner.point_cloud_running = aligner.init_point_cloud
        aligner.export(OUT_FILE)

    Path(TIME_FILE).write_text(f"{int(60 - np.round(end_time - start_time))}")