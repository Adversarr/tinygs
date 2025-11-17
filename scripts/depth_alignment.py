from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from camera import (
    CameraExtrinsic,
    CameraIntrinsic,
    calc_c2w,
    calc_w2c,
    project_points_to_camera,
)
from sklearn.linear_model import LinearRegression, RANSACRegressor

@dataclass
class DepthAlignmentResult:
    """Result container for depth alignment"""

    scale: float
    offset: float
    inlier_mask: np.ndarray
    ransac_model: RANSACRegressor
    relative_depth_values: np.ndarray
    inverse_depth_values: np.ndarray
    score: float


class DepthAlignment:
    """Class for aligning relative depth maps to actual 3D point clouds using RANSAC"""

    def __init__(self, max_trials: int = 100, random_state: int = 42, stop_score=0.99):
        """
        Initialize the depth alignment class.

        Args:
            max_trials: Maximum number of RANSAC iterations
            random_state: Random seed for reproducibility
        """
        self.max_trials = max_trials
        self.random_state = random_state
        self.ransac_model = RANSACRegressor(
            max_trials=max_trials,
            random_state=random_state,
            stop_score=stop_score,
            # loss='squared_error',
        )

    def estimate_scale(
        self,
        points: np.ndarray,
        camera_intrinsics: CameraIntrinsic,
        camera_extrinsic: CameraExtrinsic,
        relative_depth_map: np.ndarray,
    ) -> DepthAlignmentResult:
        """
        Estimate scale between relative depth map and actual 3D points using RANSAC.

        Args:
            points: 3D points in world coordinates (N, 3)
            camera_intrinsics: Camera intrinsic parameters
            camera_extrinsic: Camera extrinsic parameters for the current view
            relative_depth_map: Relative depth map from Depth Anything (H, W), normalized [0, 1]

        Returns:
            DepthAlignmentResult containing scale, offset, inlier mask, and other results
        """
        # Get transformation matrix from world to camera
        T_wc = calc_w2c(camera_extrinsic)

        # Project points to camera and filter visible points
        u_in, v_in, z_in, mask_zpos = project_points_to_camera(
            points, camera_intrinsics, T_wc
        )
        dist_to_boundaries = np.minimum(
            np.minimum(u_in, v_in),
            np.minimum(camera_intrinsics.width - u_in, camera_intrinsics.height - v_in),
        )

        # give center points higher weight
        weights = np.clip(dist_to_boundaries / np.max(dist_to_boundaries), 0.1, 1.0)

        # Sample relative depth values at projected pixel coordinates
        depth_at_proj = relative_depth_map[v_in.astype(np.int32), u_in.astype(np.int32)]

        # Inverse of actual depth (1/z)
        inv_z_in = 1.0 / (z_in + 1e-8)

        # Prepare data for RANSAC: X = relative depth, Y = inverse depth
        X = depth_at_proj.reshape(-1, 1)
        Y = inv_z_in

        # Fit RANSAC model
        self.ransac_model.fit(X, Y, sample_weight=weights)

        # Extract model parameters
        scale = self.ransac_model.estimator_.coef_[0]
        offset = self.ransac_model.estimator_.intercept_
        inliers = self.ransac_model.inlier_mask_

        # Calculate RANSAC score based on inlier ratio
        ransac_score = self.ransac_model.score(X[inliers], Y[inliers])

        # Create result object
        result = DepthAlignmentResult(
            scale=scale,
            offset=offset,
            inlier_mask=inliers,
            ransac_model=self.ransac_model,
            relative_depth_values=depth_at_proj,
            inverse_depth_values=inv_z_in,
            score=ransac_score,
        )

        return result

    def convert_relative_to_absolute_depth(
        self,
        relative_depth: np.ndarray,
        scale: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert relative depth values to absolute depth using the estimated scale.

        Args:
            relative_depth: Relative depth values (normalized [0, 1])
            scale: Scale parameter (if None, uses the last fitted model)
            offset: Offset parameter (if None, uses the last fitted model)

        Returns:
            Absolute depth values in meters
        """
        if scale is None or offset is None:
            raise ValueError("RANSAC model not fitted yet. Call estimate_scale first.")

        # Convert relative depth to inverse depth using the linear model
        inverse_depth = scale * relative_depth + offset

        # Convert inverse depth to absolute depth
        absolute_depth = 1.0 / (inverse_depth + 1e-8)

        return absolute_depth

    def unproject_depth_map(
        self,
        relative_depth_map: np.ndarray,
        camera_intrinsics: CameraIntrinsic,
        camera_extrinsic: CameraExtrinsic,
        downsampling_factor: int = 1,
        scale: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Unproject a relative depth map to 3D world coordinates.

        Args:
            relative_depth_map: Relative depth map (H, W), normalized [0, 1]
            camera_intrinsics: Camera intrinsic parameters
            camera_extrinsic: Camera extrinsic parameters
            downsampling_factor: Factor to downsample the depth map for efficiency
            scale: Scale parameter (if None, uses the last fitted model)
            offset: Offset parameter (if None, uses the last fitted model)

        Returns:
            3D points in world coordinates (N, 3)
            mask: Boolean mask indicating valid points (N,)
        """
        # Downsample the depth map
        depth_down = relative_depth_map[::downsampling_factor, ::downsampling_factor]
        h_d, w_d = depth_down.shape

        # Create pixel coordinates for the downsampled depth map
        v_coords, u_coords = np.meshgrid(np.arange(h_d), np.arange(w_d), indexing="ij")
        u_coords = u_coords * downsampling_factor  # Scale back to original resolution
        v_coords = v_coords * downsampling_factor

        # Flatten the coordinates and depth
        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        depth_flat = depth_down.flatten()
        depth_std = np.std(depth_flat)
        # Filter outliers
        mask = np.abs(depth_flat - np.mean(depth_flat)) < 3 * depth_std
        mask = mask & (depth_flat > 0)
        # mask = depth_flat > 0
        u_flat = u_flat[mask]
        v_flat = v_flat[mask]
        depth_flat = depth_flat[mask]

        # Convert relative depth to absolute depth
        actual_depth = self.convert_relative_to_absolute_depth(
            depth_flat, scale, offset
        )

        # Unproject to 3D camera coordinates
        x_cam = (u_flat - camera_intrinsics.cx) * actual_depth / camera_intrinsics.fx
        y_cam = (v_flat - camera_intrinsics.cy) * actual_depth / camera_intrinsics.fy
        z_cam = actual_depth

        # Stack into 3D points in camera coordinates
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Get camera-to-world transformation matrix
        T_cw = calc_c2w(camera_extrinsic)
        R_cw = T_cw[0:3, 0:3]
        t_cw = T_cw[0:3, 3]

        # Transform to world coordinates
        points_world = (R_cw @ points_cam.T).T + t_cw

        return points_world, mask

    def restore(self, result: DepthAlignmentResult):
        self.scale = result.scale
        self.offset = result.offset
        self.ransac_model = result.ransac_model
