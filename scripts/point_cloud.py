from dataclasses import dataclass
from warnings import warn

import numpy as np
import open3d as o3d
from camera import load_point_cloud

try:
    import pcloudsim
except ImportError:
    pcloudsim = None
    warn("pcloudsim not found, point cloud operations maybe slow.")

@dataclass
class PointCloudUpdateConfig:
    # statscical outlier removal
    enable_statistical_outlier_removal: bool = True
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 2.0

    # radius outlier removal
    enable_radius_outlier_removal: bool = True
    ror_nb_points: int = 16
    ror_radius: float = 0.05

    # voxel downsampling
    enable_voxel_downsampling: bool = True
    ds_voxel_size: float = 0.008

class PointCloudManipulator:
    def __init__(self, init_xyz: np.ndarray, init_rgb: np.ndarray) -> None:
        self.pcd = o3d.geometry.PointCloud()

        self.pcd.points = o3d.utility.Vector3dVector(init_xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(init_rgb)
        self.init_mean = np.mean(init_xyz, axis=0) # [3, ]

    def add_points(self, new_xyz: np.ndarray, new_rgb: np.ndarray) -> None:
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_xyz)
        new_pcd.colors = o3d.utility.Vector3dVector(new_rgb)

        self.pcd += new_pcd

    def update(self, config: PointCloudUpdateConfig) -> None:
        if pcloudsim is not None:
            params = pcloudsim.RemovalParams()
            params.enable_statistical_outliers = config.enable_statistical_outlier_removal
            params.n_neighbors_stats = config.sor_nb_neighbors
            params.std_dev_mul = config.sor_std_ratio

            params.enable_radius_outliers = config.enable_radius_outlier_removal
            params.radius = config.ror_radius
            params.min_points_radius = config.ror_nb_points

            params.enable_voxel_simplify = config.enable_voxel_downsampling
            params.voxel_size = config.ds_voxel_size

            points = np.asarray(self.pcd.points, dtype=np.float32)
            colors = np.asarray(self.pcd.colors, dtype=np.float32)

            points, colors = pcloudsim.simplify_point_cloud(points, colors, params)

            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            return
        else: # Slow path
            if config.enable_statistical_outlier_removal:
                cl, ind = self.pcd.remove_statistical_outlier(
                    nb_neighbors=config.sor_nb_neighbors,
                    std_ratio=config.sor_std_ratio
                )
                self.pcd = self.pcd.select_by_index(ind)

            if config.enable_radius_outlier_removal:
                cl, ind = self.pcd.remove_radius_outlier(
                    nb_points=config.ror_nb_points,
                    radius=config.ror_radius
                )
                self.pcd = self.pcd.select_by_index(ind)

            if config.enable_voxel_downsampling:
                self.pcd = self.pcd.voxel_down_sample(voxel_size=config.ds_voxel_size)

    def clone(self) -> 'PointCloudManipulator':
        return PointCloudManipulator(
            np.asarray(self.pcd.points).copy(),
            np.asarray(self.pcd.colors).copy()
        )

    @staticmethod
    def load_from_file(file_path: str) -> 'PointCloudManipulator':
        ids, xyz, rgb = load_point_cloud(file_path)
        return PointCloudManipulator(xyz, rgb)

    def save_to_file(self, file_path: str) -> None:
        o3d.io.write_point_cloud(file_path, self.pcd)


    @property
    def mean(self) -> np.ndarray:
        return np.mean(np.asarray(self.pcd.points), axis=0)

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray(self.pcd.points)

    @property
    def rgb(self) -> np.ndarray:
        return np.asarray(self.pcd.colors)
