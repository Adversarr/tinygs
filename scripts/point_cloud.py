import open3d as o3d
import numpy as np
from dataclasses import dataclass

from camera import load_point_cloud

@dataclass
class PointCloudUpdateConfig:
    # statscical outlier removal
    enable_statistical_outlier_removal: bool = True
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 2.0

    # radius outlier removal
    enable_radius_outlier_removal: bool = False
    ror_nb_points: int = 16
    ror_radius: float = 0.05

    # voxel downsampling
    enable_voxel_downsampling: bool = True
    ds_voxel_size: float = 0.02


class PointCloudManipulator:
    def __init__(self, init_xyz: np.ndarray, init_rgb: np.ndarray) -> None:
        self.pcd = o3d.geometry.PointCloud()

        self.pcd.points = o3d.utility.Vector3dVector(init_xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(init_rgb)

    def add_points(self, new_xyz: np.ndarray, new_rgb: np.ndarray) -> None:
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_xyz)
        new_pcd.colors = o3d.utility.Vector3dVector(new_rgb)

        self.pcd += new_pcd

    def update(self, config: PointCloudUpdateConfig) -> None:
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