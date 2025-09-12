import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from typing import Tuple, Optional
from dataclasses import dataclass
from camera import CameraIntrinsic, CameraExtrinsic, calc_c2w, project_points_to_camera, calc_w2c


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
    
    def __init__(self, max_trials: int = 1000, random_state: int = 42):
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
            random_state=random_state
        )
    
    def estimate_scale(
        self,
        points: np.ndarray,
        point_ids: np.ndarray,
        camera_intrinsics: CameraIntrinsic,
        camera_extrinsic: CameraExtrinsic,
        relative_depth_map: np.ndarray
    ) -> DepthAlignmentResult:
        """
        Estimate scale between relative depth map and actual 3D points using RANSAC.
        
        Args:
            points: 3D points in world coordinates (N, 3)
            point_ids: IDs corresponding to the points
            camera_intrinsics: Camera intrinsic parameters
            camera_extrinsic: Camera extrinsic parameters for the current view
            relative_depth_map: Relative depth map from Depth Anything (H, W), normalized [0, 1]
            
        Returns:
            DepthAlignmentResult containing scale, offset, inlier mask, and other results
        """
        # Get transformation matrix from world to camera
        T_wc = calc_w2c(camera_extrinsic)
        
        # Project points to camera and filter visible points
        u_in, v_in, z_in, pids_in, mask_zpos = project_points_to_camera(
            points, point_ids, camera_intrinsics, T_wc
        )
        
        # Sample relative depth values at projected pixel coordinates
        depth_at_proj = relative_depth_map[
            v_in.astype(np.int32), 
            u_in.astype(np.int32)
        ]
        
        # Inverse of actual depth (1/z)
        inv_z_in = 1.0 / (z_in + 1e-8)
        
        # Prepare data for RANSAC: X = relative depth, Y = inverse depth
        X = depth_at_proj.reshape(-1, 1)
        Y = inv_z_in
        
        # Fit RANSAC model
        self.ransac_model.fit(X, Y)
        
        # Extract model parameters
        scale = self.ransac_model.estimator_.coef_[0]
        offset = self.ransac_model.estimator_.intercept_

        # Calculate RANSAC score based on inlier ratio
        ransac_score = self.ransac_model.score(X, Y)

        # Create result object
        result = DepthAlignmentResult(
            scale=scale,
            offset=offset,
            inlier_mask=self.ransac_model.inlier_mask_,
            ransac_model=self.ransac_model,
            relative_depth_values=depth_at_proj,
            inverse_depth_values=inv_z_in,
            score=ransac_score
        )
        
        return result
    
    def convert_relative_to_absolute_depth(
        self,
        relative_depth: np.ndarray,
        scale: Optional[float] = None,
        offset: Optional[float] = None
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
        offset: Optional[float] = None
    ) -> np.ndarray:
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
        """
        # Downsample the depth map
        depth_down = relative_depth_map[::downsampling_factor, ::downsampling_factor]
        h_d, w_d = depth_down.shape
        
        # Create pixel coordinates for the downsampled depth map
        v_coords, u_coords = np.meshgrid(np.arange(h_d), np.arange(w_d), indexing='ij')
        u_coords = u_coords * downsampling_factor  # Scale back to original resolution
        v_coords = v_coords * downsampling_factor
        
        # Flatten the coordinates and depth
        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        depth_flat = depth_down.flatten()
        
        # Convert relative depth to absolute depth
        actual_depth = self.convert_relative_to_absolute_depth(depth_flat, scale, offset)
        
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
        
        return points_world


class DepthAlignment_v2:
    """Depth alignment variant that fits inverse(relative_depth) -> z (actual depth) using RANSAC."""
    def __init__(self, max_trials: int = 1000, random_state: int = 42):
        """
        Initialize the DepthAlignment_v2 instance.
        Args:
            max_trials: Maximum number of RANSAC iterations
            random_state: Random seed for reproducibility
        """
        self.max_trials = max_trials
        self.random_state = random_state
        self.ransac_model = RANSACRegressor(
            max_trials=max_trials,
            random_state=random_state
        )

    def estimate_scale(
        self,
        points: np.ndarray,
        point_ids: np.ndarray,
        camera_intrinsics: CameraIntrinsic,
        camera_extrinsic: CameraExtrinsic,
        relative_depth_map: np.ndarray
    ) -> DepthAlignmentResult:
        """
        Estimate linear mapping from inverse(relative_depth) to actual depth z using RANSAC.

        X = 1 / relative_depth,  Y = z_in
        """
        # World->camera transform
        T_wc = calc_w2c(camera_extrinsic)

        # Project points to camera and filter visible points
        u_in, v_in, z_in, pids_in, mask_zpos = project_points_to_camera(
            points, point_ids, camera_intrinsics, T_wc
        )

        # Sample relative depth at projected pixel coordinates
        depth_at_proj = relative_depth_map[
            v_in.astype(np.int32),
            u_in.astype(np.int32)
        ]

        # Inverse of relative depth (1 / d_rel)
        inv_rel = 1.0 / (depth_at_proj + 1e-8)

        # Prepare data for RANSAC: X = inv_rel, Y = z_in
        X = inv_rel.reshape(-1, 1)
        Y = z_in

        # Fit RANSAC model
        self.ransac_model.fit(X, Y)

        # Extract model parameters (z = scale * inv_rel + offset)
        scale = float(self.ransac_model.estimator_.coef_[0])
        offset = float(self.ransac_model.estimator_.intercept_)

        # Calculate RANSAC score
        ransac_score = self.ransac_model.score(X, Y)

        result = DepthAlignmentResult(
            scale=scale,
            offset=offset,
            inlier_mask=self.ransac_model.inlier_mask_,
            ransac_model=self.ransac_model,
            relative_depth_values=depth_at_proj,
            inverse_depth_values=z_in,
            score=ransac_score
        )

        return result

    def convert_relative_to_absolute_depth(
        self,
        relative_depth: np.ndarray,
        scale: Optional[float] = None,
        offset: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert relative depth values to absolute depth (z) using the learned linear model on inverse(relative_depth).

        Model: z = scale * (1 / relative_depth) + offset
        """
        if scale is None or offset is None:
            raise ValueError("RANSAC model not fitted yet. Call estimate_scale first.")

        inv_rel = 1.0 / (relative_depth + 1e-8)
        predicted_z = scale * inv_rel + offset
        return predicted_z

    def unproject_depth_map(
        self,
        relative_depth_map: np.ndarray,
        camera_intrinsics: CameraIntrinsic,
        camera_extrinsic: CameraExtrinsic,
        downsampling_factor: int = 1,
        scale: Optional[float] = None,
        offset: Optional[float] = None
    ) -> np.ndarray:
        """
        Unproject relative depth map to 3D world coordinates using the v2 conversion (inverse relative -> z).
        """
        # Downsample the depth map
        depth_down = relative_depth_map[::downsampling_factor, ::downsampling_factor]
        h_d, w_d = depth_down.shape

        # Pixel coordinates for downsampled map
        v_coords, u_coords = np.meshgrid(np.arange(h_d), np.arange(w_d), indexing='ij')
        u_coords = u_coords * downsampling_factor
        v_coords = v_coords * downsampling_factor

        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        depth_flat = depth_down.flatten()

        # Convert relative depth to actual depth z using the v2 model
        actual_depth = self.convert_relative_to_absolute_depth(depth_flat, scale, offset)

        # Unproject to camera coordinates
        x_cam = (u_flat - camera_intrinsics.cx) * actual_depth / camera_intrinsics.fx
        y_cam = (v_flat - camera_intrinsics.cy) * actual_depth / camera_intrinsics.fy
        z_cam = actual_depth

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Camera-to-world transform
        T_cw = calc_c2w(camera_extrinsic)
        R_cw = T_cw[0:3, 0:3]
        t_cw = T_cw[0:3, 3]

        points_world = (R_cw @ points_cam.T).T + t_cw

        return points_world