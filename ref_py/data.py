import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from PIL import Image
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

# Constants
DEFAULT_T_FACTOR = 10
SUPPORTED_VIDEO_EXTENSIONS = [".mp4"]
SUPPORTED_CAMERA_MODEL = "PINHOLE"
IMAGE_FORMAT = "png"
RGB_NORMALIZATION_FACTOR = 255.0

# File paths
SLAM_DIR = "inputs/slam"
CAMERAS_FILE = "cameras.txt"
POINTS3D_FILE = "points3D.txt"
TRAJECTORY_FILE = "inputs/traj_full.txt.bak"




def _parse_camera_ext(images_file: Path) -> Tuple[List, List]:
    """Parse camera extrinsics from trajectory file.
    
    Args:
        images_file: Path to trajectory file containing camera poses
        
    Returns:
        Tuple of (w2c_mats, c2w_mats) - world-to-camera and camera-to-world matrices
    """
    with open(images_file, "r") as f:
        # id qw qx qy qz tx ty tz imgid T_cw
        w2c_mats = []
        c2w_mats = []
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            data = line.split()
            qw, qx, qy, qz = map(float, data[1:5])
            tx, ty, tz = map(float, data[5:8])
            rot = Rotation.from_quat([qx, qy, qz, qw])
            R_mat = rot.as_matrix()
            bottom = np.array([[0, 0, 0, 1]])
            t_vec = np.array([tx, ty, tz])
            w2c = np.concatenate([np.concatenate([R_mat, t_vec.reshape(3, -1)], 1), bottom], axis=0)
            c2w = np.linalg.inv(w2c)
            w2c_mats.append(w2c)
            c2w_mats.append(c2w)
    return w2c_mats, c2w_mats


@dataclass
class Camera:
    """Camera intrinsic parameters from COLMAP format."""
    id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float


def _parse_sfm_points(sfm_points_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse SfM 3D points from COLMAP points3D.txt file.
    
    Args:
        sfm_points_file: Path to points3D.txt file
        
    Returns:
        Tuple of (xyz, rgb) - 3D coordinates and RGB colors (normalized to [0,1])
    """
    points_xyz_rgb = np.loadtxt(sfm_points_file, usecols=(1, 2, 3, 4, 5, 6), dtype=np.float32)
    print(f"Loaded {points_xyz_rgb.shape[0]} 3D points.")

    xyz = points_xyz_rgb[:, :3]
    rgb = points_xyz_rgb[:, 3:] / RGB_NORMALIZATION_FACTOR
    return np.array(xyz), np.array(rgb)


def _get_camera_intrinsics(camera: Camera) -> np.ndarray:
    """Convert Camera object to 3x3 intrinsics matrix K.
    
    Args:
        camera: Camera object with intrinsic parameters
        
    Returns:
        3x3 camera intrinsics matrix
    """
    return np.array(
        [
            [camera.fx, 0, camera.cx],
            [0, camera.fy, camera.cy],
            [0, 0, 1],
        ]
    )


def _parse_camera_int(data_dir: Path) -> Camera:
    """Parse camera intrinsics from COLMAP cameras.txt file.
    
    Args:
        data_dir: Root directory containing inputs/slam/cameras.txt
        
    Returns:
        Camera object with parsed intrinsic parameters
    """
    with open(data_dir / SLAM_DIR / CAMERAS_FILE, "r") as f:
        # CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY K1 K2 K3 P1 P2
        # there should be exactly 1 camera
        line = f.readline()
        data = line.strip().split()
        camera = Camera(
            id=int(data[0]),
            model=data[1],
            width=int(data[2]),
            height=int(data[3]),
            fx=float(data[4]),
            fy=float(data[5]),
            cx=float(data[6]),
            cy=float(data[7]),
            k1=float(data[8]),
            k2=float(data[9]),
            k3=float(data[10]),
            p1=float(data[11]),
            p2=float(data[12]),
        )
        return camera


def _extract_original_images(
    data_dir: Path, image_dir: Path, T_factor: int, width: int, height: int
) -> None:
    """Extract and resize frames from MP4 video in competition dataset.
    
    Processes the video file by:
    - Extracting every T_factor-th frame
    - Resizing frames to specified dimensions
    - Saving as PNG images with zero-padded filenames
    
    Args:
        data_dir: Directory containing the MP4 video file
        image_dir: Output directory for extracted images
        T_factor: Frame sampling rate (extract every T_factor-th frame)
        width: Target image width
        height: Target image height
    """
    candidate_videos = [
        v for v in os.listdir(data_dir) 
        if any(v.endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS)
    ]
    if len(candidate_videos) == 0:
        raise RuntimeError(f"No MP4 video found in directory: {data_dir}")
    elif len(candidate_videos) > 1:
        raise RuntimeError(
            f"Multiple videos found in {data_dir}: {candidate_videos}. "
            f"Expected exactly one MP4 file."
        )
    video = data_dir / candidate_videos[0]
    # Load mp4 file frame by frame and save the images to image_dir
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(image_dir, exist_ok=True)
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        if i % T_factor != 0:  # fxxk, we have to drop the first frame.
            continue

        resized_size = (width, height)
        resized_image = np.array(
            Image.fromarray(frame).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(image_dir / f"{i:04d}.{IMAGE_FORMAT}", resized_image)

    print(f"Extracted {frame_count} frames from {video} to {image_dir}.")


class CompetitionParser:
    """Parser for competition dataset containing video, camera poses, and SfM points.
    
    Extracts frames from MP4 video, loads camera intrinsics/extrinsics, and SfM 3D points.
    Used for 3D Gaussian Splatting training and visualization.
    
    Args:
        data_dir: Root directory containing the competition dataset
        T_factor: Frame sampling rate (extract every T_factor-th frame)
        normalize: Whether to normalize the scene using similarity transform and PCA alignment
        rebuild: Whether to rebuild the image cache
        
    Attributes:
        images: List of extracted image file paths
        c2w_mats: Camera-to-world transformation matrices
        w2c_mats: World-to-camera transformation matrices
        cam_K: 3x3 camera intrinsics matrix
        sfm_points: SfM 3D point coordinates
        sfm_colors: SfM 3D point colors
        width, height: Image dimensions
        transform: 4x4 normalization transformation matrix (identity if normalize=False)
    """
    def __init__(
        self,
        data_dir: str,
        T_factor: int = DEFAULT_T_FACTOR,
        normalize: bool = False,
        rebuild: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.scene_dir = self.data_dir / SLAM_DIR
        ### camera
        self.camera = _parse_camera_int(self.data_dir)
        self.width = self.camera.width
        self.height = self.camera.height
        self.T_factor = T_factor
        self.cam_K = _get_camera_intrinsics(self.camera)
        if self.camera.model != SUPPORTED_CAMERA_MODEL:
            raise RuntimeError(
                f"Unsupported camera model: {self.camera.model}. "
                f"Only PINHOLE camera model is supported."
            )

        self.image_dir = self.data_dir / f"inputs/images_{self.width}x{self.height}_{self.T_factor}"
        ### Make the training images.
        if rebuild:  # Clear previous cache if commanded.
            if self.image_dir.exists():
                shutil.rmtree(self.image_dir)
                print(f"Removed {self.image_dir}")
        # Extract training images.
        if not self.image_dir.exists():
            _extract_original_images(
                self.data_dir, self.image_dir, T_factor, self.width, self.height
            )
        self.images = list(self.image_dir.glob(f'*.{IMAGE_FORMAT}'))
        self.images.sort()
        print(f"Got {len(self.images)} images.")

        ### world to camera matrices.
        # self.w2c_mats, self.c2w_mats = _parse_camera_ext(self.scene_dir / "images.txt")
        self.w2c_mats, self.c2w_mats = _parse_camera_ext(self.data_dir / TRAJECTORY_FILE)
        self.w2c_mats = np.array(self.w2c_mats[::T_factor])
        self.c2w_mats = np.array(self.c2w_mats[::T_factor])
        print(f"Got {len(self.w2c_mats)} world to camera matrices.")

        ### sfm points
        self.sfm_points, self.sfm_colors = _parse_sfm_points(
            self.scene_dir / POINTS3D_FILE
        )
        self.sfm_points = self.sfm_points.astype(np.float32)
        self.sfm_colors = self.sfm_colors.astype(np.float32)
        print(
            f"Got {len(self.sfm_points)} sfm points, ranges: {self.sfm_points.min(0)} ~ {self.sfm_points.max(0)}"
        )
        
        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(self.c2w_mats)
            self.c2w_mats = transform_cameras(T1, self.c2w_mats)
            self.sfm_points = transform_points(T1, self.sfm_points)

            T2 = align_principal_axes(self.sfm_points)
            self.c2w_mats = transform_cameras(T2, self.c2w_mats)
            self.sfm_points = transform_points(T2, self.sfm_points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(self.sfm_points[:, 2]) > np.mean(self.sfm_points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                self.c2w_mats = transform_cameras(T3, self.c2w_mats)
                self.sfm_points = transform_points(T3, self.sfm_points)
                transform = T3 @ transform
                
            # Update w2c_mats to be consistent with normalized c2w_mats
            self.w2c_mats = np.linalg.inv(self.c2w_mats)
            
            self.transform = transform
            print(f"Applied normalization transform with shape {transform.shape}")
            print(f"Normalized sfm points ranges: {self.sfm_points.min(0)} ~ {self.sfm_points.max(0)}")
        else:
            self.transform = np.eye(4)
        
        # # TODO: undistortion, but does not apply to PINHOLE?
        # self.K_undist, self.roi_undist = cv2.getOptimalNewCameraMatrix(
        #     self.cam_K, np.empty(0, dtype=np.float32), (self.width, self.height), 0
        # )
        # self.map_x, self.map_y = cv2.initUndistortRectifyMap(
        #     self.cam_K, np.empty(0, dtype=np.float32), None, self.K_undist, (self.width, self.height), cv2.CV_32FC1
        # )

    def __getitem__(self, item: int) -> np.ndarray:
        """Load and normalize image by index.
        
        Args:
            item: Image index
            
        Returns:
            Normalized image as float32 array in range [0,1]
        """
        # uint8 -> float32
        return imageio.imread(self.images[item]).astype(np.float32) / RGB_NORMALIZATION_FACTOR

class CompetitionDataset(Dataset):
    """PyTorch Dataset wrapper for CompetetionParser.
    
    Provides batched access to images, camera matrices, and intrinsics for training.
    
    Args:
        parser: CompetitionParser instance
        split: Dataset split (currently unused, defaults to "train")
    """
    def __init__(self, parser: CompetitionParser, split: str = "train"):
        super().__init__()
        self.parser = parser
        self.split = split

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return min(len(self.parser.w2c_mats), len(self.parser.images))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """Get dataset item by index.
        
        Args:
            item: Sample index
            
        Returns:
            Dictionary containing:
                - K: Camera intrinsics matrix [3, 3]
                - image: Normalized image tensor [H, W, 3]
                - camtoworld: Camera-to-world matrix [4, 4]
                - w2c: World-to-camera matrix [4, 4]
        """
        image = self.parser[item]
        c2w_mat = self.parser.c2w_mats[item]
        w2c_mat = self.parser.w2c_mats[item]
        K = self.parser.cam_K.copy()

        return {
            "K": torch.from_numpy(K).float(), # [3, 3]
            "image": torch.from_numpy(image).float(),
            "camtoworld": torch.from_numpy(c2w_mat).float(),
            'w2c': torch.from_numpy(w2c_mat).float(),
        }

if __name__ == "__main__":
    folder = "/data/accgs/1747834320424/"
    parser = CompetitionParser(folder)
    dataset = CompetitionDataset(parser)
    item = dataset[0]
    for k, v in item.items():
        print(k, v.shape)
