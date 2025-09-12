import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R


@dataclass
class CameraIntrinsic:
    """Camera intrinsic parameters"""
    width: int
    height: int
    fx: float  # focal length x
    fy: float  # focal length y
    cx: float  # principal point x
    cy: float  # principal point y


@dataclass
class CameraExtrinsic:
    """Camera extrinsic parameters"""
    img_id: int  # image ID
    rotation: np.ndarray  # quaternion [w, x, y, z]
    translation: np.ndarray  # translation [x, y, z]


def load_camera_intrinsics(file_path: str) -> CameraIntrinsic:
    """Load camera intrinsics from COLMAP cameras.txt file.
    
    Args:
        file_path: Path to the cameras.txt file
        
    Returns:
        CameraIntrinsic object with parsed parameters
        
    Raises:
        RuntimeError: If parsing fails or unsupported camera model
    """
    with open(file_path, 'r') as f:
        cam_w = cam_h = fx = fy = cx = cy = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # parts: [camera_id, model, width, height, params...]
            cam_id = int(parts[0])
            model = parts[1]
            cam_w = int(parts[2])
            cam_h = int(parts[3])
            params = list(map(float, parts[4:]))
            if model.upper().startswith('PINHOLE'):
                # PINHOLE: fx fy cx cy ...
                if len(params) < 4:
                    raise RuntimeError("Expected at least 4 params for PINHOLE")
                fx, fy, cx, cy = params[0:4]
            elif model.upper().startswith('SIMPLE_PINHOLE'):
                # SIMPLE_PINHOLE: f cx cy
                if len(params) < 3:
                    raise RuntimeError("Expected at least 3 params for SIMPLE_PINHOLE")
                f, cx, cy = params[0:3]
                fx = fy = f
            else:
                # 兜底：如果参数数量足够，尝试提取 fx,fy,cx,cy
                if len(params) >= 4:
                    fx, fy, cx, cy = params[0:4]
                else:
                    raise RuntimeError(f"Unsupported camera model '{model}' or insufficient params")
            break

    if None in (cam_w, cam_h, fx, fy, cx, cy):
        raise RuntimeError("Failed to parse camera intrinsics")
        
    return CameraIntrinsic(width=cam_w, height=cam_h, fx=fx, fy=fy, cx=cx, cy=cy)


def load_camera_extrinsics(file_path: str) -> Tuple[List[CameraExtrinsic], np.ndarray, np.ndarray]:
    """Load camera extrinsics from COLMAP images.txt file.
    
    Args:
        file_path: Path to the images.txt file
        
    Returns:
        Tuple of (list of CameraExtrinsic objects, quaternions array, translations array)
        
    Raises:
        RuntimeError: If no camera poses are parsed
    """
    qs = []
    ts = []
    img_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # images.txt 每张图片的第一行通常包含: image_id qw qx qy qz tx ty tz camera_id name
            # 尝试解析 7 个 float 项（从 parts[1] 到 parts[7]）
            if len(parts) >= 8:
                try:
                    img_id = int(parts[0])  # image_id
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    qs.append([qw, qx, qy, qz])
                    ts.append([tx, ty, tz])
                    img_ids.append(img_id)
                except ValueError:
                    # 若解析失败则跳过（可能是匹配行而非图像行）
                    continue

    if len(qs) == 0:
        raise RuntimeError("No camera poses parsed from images.txt")

    qs = np.array(qs, dtype=np.float64)
    ts = np.array(ts, dtype=np.float64)
    
    # Create CameraExtrinsic objects
    extrinsics = []
    for i in range(len(qs)):
        extrinsics.append(CameraExtrinsic(
            img_id=img_ids[i],
            rotation=qs[i],
            translation=ts[i]
        ))
    
    return extrinsics, qs, ts


def load_point_cloud(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load point cloud data from COLMAP points3D.txt file.
    
    Args:
        file_path: Path to the points3D.txt file
        
    Returns:
        Tuple of (point IDs array, points array, colors array)
    """
    point_ids = []
    points = []
    colors = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # points3D.txt 格式: pointId X Y Z R G B error track...
            try:
                pid = int(parts[0])
                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])
                point_ids.append(pid)
                points.append([x, y, z])
                colors.append([r, g, b])
            except:
                continue

    point_ids = np.array(point_ids, dtype=np.int32)
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8)
    
    return point_ids, points, colors


def calc_c2w(extrinsic: CameraExtrinsic) -> np.ndarray:
    """Convert camera extrinsic parameters to transformation matrix T_cw.
    
    This function creates a 4x4 transformation matrix that converts from camera
    coordinates to world coordinates using the quaternion and translation from
    the CameraExtrinsic object.
    
    Args:
        extrinsic: CameraExtrinsic object containing rotation (quaternion) and translation
        
    Returns:
        4x4 transformation matrix T_cw that converts camera coordinates to world coordinates
    """
    # Extract quaternion and translation
    qw, qx, qy, qz = extrinsic.rotation
    tx, ty, tz = extrinsic.translation
    
    # Convert quaternion to rotation matrix
    quat_scipy = np.array([qx, qy, qz, qw], dtype=np.float64)
    R_wc = R.from_quat(quat_scipy).as_matrix()
    t_wc = np.array([tx, ty, tz], dtype=np.float64)
    
    # Create camera-to-world transformation matrix
    T_cw = np.eye(4)
    T_cw[0:3, 0:3] = R_wc.T  # Transpose of rotation matrix
    T_cw[0:3, 3] = -R_wc.T @ t_wc  # Translation component
    
    return T_cw


def calc_w2c(extrinsic: CameraExtrinsic) -> np.ndarray:
    """Convert camera extrinsic parameters to transformation matrix T_wc.
    
    This function creates a 4x4 transformation matrix that converts from world
    coordinates to camera coordinates using the quaternion and translation from
    the CameraExtrinsic object.
    
    Args:
        extrinsic: CameraExtrinsic object containing rotation (quaternion) and translation
        
    Returns:
        4x4 transformation matrix T_wc that converts world coordinates to camera coordinates
    """
    # Extract quaternion and translation
    qw, qx, qy, qz = extrinsic.rotation
    tx, ty, tz = extrinsic.translation
    
    # Convert quaternion to rotation matrix
    quat_scipy = np.array([qx, qy, qz, qw], dtype=np.float64)
    R_wc = R.from_quat(quat_scipy).as_matrix()
    t_wc = np.array([tx, ty, tz], dtype=np.float64)
    
    # Create world-to-camera transformation matrix
    T_wc = np.eye(4)
    T_wc[0:3, 0:3] = R_wc  # Rotation matrix
    T_wc[0:3, 3] = t_wc    # Translation component
    
    return T_wc


def project_points_to_camera(points: np.ndarray, point_ids: np.ndarray,
                            camera_intrinsics: CameraIntrinsic,
                            T_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to camera coordinates and filter visible points.
    
    This function projects 3D world points to camera coordinates, filters points
    that are in front of the camera (z > 0), and further filters points that
    fall within the camera's image bounds.
    
    Args:
        points: Array of 3D points in world coordinates (N, 3)
        point_ids: Array of point IDs corresponding to the points
        camera_intrinsics: Camera intrinsic parameters
        T_wc: World-to-camera transformation matrix (4x4)
        
    Returns:
        Tuple of (u coordinates, v coordinates, z depths, point IDs, unified_mask) for points
        that are both in front of the camera and within image bounds
    """
    # Extract rotation matrix and translation vector from T_wc
    R_wc = T_wc[0:3, 0:3]  # Rotation matrix from world to camera
    t_wc = T_wc[0:3, 3]    # Translation vector from world to camera
    
    # COLMAP convention: X_cam = R * X_world + t
    points_cam = (R_wc @ points.T).T + t_wc  # shape (N,3)
    
    x_c = points_cam[:, 0]
    y_c = points_cam[:, 1]
    z_c = points_cam[:, 2]
    
    # Only keep points in front of the camera (z > 0)
    mask_zpos = z_c > 0
    print(f"Points with z>0: {mask_zpos.sum()} / {points.shape[0]}")
    
    x_c = x_c[mask_zpos]
    y_c = y_c[mask_zpos]
    z_c = z_c[mask_zpos]
    pids_visible = point_ids[mask_zpos]
    
    # Project to pixel plane: u = fx * (x/z) + cx ; v = fy * (y/z) + cy
    u = camera_intrinsics.fx * (x_c / z_c) + camera_intrinsics.cx
    v = camera_intrinsics.fy * (y_c / z_c) + camera_intrinsics.cy
    
    # Only keep points that fall within image bounds
    mask_in_img = (u >= 0) & (u < camera_intrinsics.width) & (v >= 0) & (v < camera_intrinsics.height)
    print(f"Points inside image bounds: {mask_in_img.sum()} / {mask_zpos.sum()}")
    
    u_in = u[mask_in_img]
    v_in = v[mask_in_img]
    z_in = z_c[mask_in_img]
    pids_in = pids_visible[mask_in_img]
    
    # Create unified mask for the original points array
    unified_mask = np.zeros(points.shape[0], dtype=bool)
    # Get indices of points that passed both filters
    visible_indices = np.where(mask_zpos)[0]
    final_indices = visible_indices[mask_in_img]
    unified_mask[final_indices] = True
    
    return u_in, v_in, z_in, pids_in, unified_mask

def apply_transformation(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transformation matrix to 3D points.
    
    Args:
        points: Array of 3D points (N, 3)
        T: 4x4 transformation matrix
        
    Returns:
        Transformed 3D points (N, 3)
    """
    N = points.shape[0]
    points_hom = np.hstack((points, np.ones((N, 1), dtype=points.dtype)))  # Convert to homogeneous coordinates
    points_transformed_hom = (T @ points_hom.T).T  # Apply transformation
    points_transformed = points_transformed_hom[:, 0:3] / points_transformed_hom[:, 3:4]  # Convert back to Cartesian coordinates
    return points_transformed