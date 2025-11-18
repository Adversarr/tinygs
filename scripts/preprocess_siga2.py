"""
It preprocess the raw result from SIGA2 dataset.

Each scene should contain the following (example names shown):
```
|-- DATA_ROOT
    |-- <scene_name>/
            |-- images/   # Downsampled GT image sequence
            |-- images_gt_downsampled/   # Downsampled GT image sequence, also valid, prefer images
            |-- sparse/                 # Camera parameters (official)
            |-- train_test_split.json    # Train/test view split
```

```
{
    "train": [
        "000000.png",
        "000001.png",
        "000002.png",
        "000003.png",
        "000004.png",
        "000005.png",
        "000006.png",
        "000008.png"
    ]
}
```

Your code should load data and cameras from this structure and strictly follow `train_test_split.json` to separate training and testing.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import shutil
import trimesh
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to your root of SIGA2 dataset.')
    parser.add_argument("--id", type=str, required=True, help='ID of the scene you want to preprocess.')
    parser.add_argument('--output', type=str, required=True, help='Tempoerary output directory for training/evaluation.')
    parser.add_argument('--magic-number', type=int, default=42, help='Magic number to prepend to the UUID of each image.')
    parser.add_argument("--normalize", action='store_true', help='Normalize points3D to normal distributed.')
    args = parser.parse_args()
    assert args.magic_number > 0, f"❌ Magic number must be positive to ensure unique UUID, but got {args.magic_number}."
    return args

def readlines_and_prune(file_path: Path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # no pre- post- \s
    lines = [line.strip() for line in lines]
    # no empty lines or commented
    lines = [line for line in lines if line != '' and not line.startswith('#')]
    return lines

def parse_first_camera(file_of_camera_intrisic):
    """
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 1
    1 PINHOLE 1214 1618 1258.7539323359774 1272.4770554189026 607 809
    
    Interpreted as:
    ID MODEL WIDTH HEIGHT FX FY CX CY
    """
    lines = readlines_and_prune(file_of_camera_intrisic)
    assert len(lines) == 1, f"❌ Expected 1 camera, but got {len(lines)}."
    line = lines[0]
    camera_id, model, width, height, fx, fy, cx, cy = line.split()
    assert model == 'PINHOLE', f"❌ Expected PINHOLE model, but got {model}."
    try:
        return {
            'id': int(camera_id),
            'model': model,
            'width': int(width),
            'height': int(height),
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
        }
    except ValueError as e:
        raise ValueError(f"❌ Failed to parse camera intrinsics: {e}") from e

def parse_single_pose(line: str) -> dict:
    """
    Parse a single pose line.

    Returns:
        dict: A dictionary containing the parsed pose information.
    """

    frame_id, rig_id, qw, qx, qy, qz, tx, ty, tz, num_data_ids, *_ = line.split()
    assert int(num_data_ids) == 1, f"❌ Expected 1 data ID, but got {num_data_ids}."
    try:
        return {
            'frame_id': int(frame_id) - 1, # Frame ID is 1-indexed, but we want 0-indexed.
            'rig_id': int(rig_id),
            'qw': float(qw),
            'qx': float(qx),
            'qy': float(qy),
            'qz': float(qz),
            'tx': float(tx),
            'ty': float(ty),
            'tz': float(tz),
        }
    except ValueError as e:
        raise ValueError(f"❌ Failed to parse pose: {e}") from e


def parse_poses(file_of_poses) -> dict[int, dict]:
    """
    # Frame list with one line of data per frame:
    #   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)
    1 1 0.99327519673104903 0.03583486016365197 0.088061502244715609 0.066071311310971242 -0.8567568908854013 -0.80960094575852626 1.4537642680930103 1 CAMERA 1 1
    2 1 0.99332061988404452 0.03554105551421452 0.087965961307444454 0.065673199536571594 -0.85684555212765534 -0.81016635795194825 1.4558832797791708 1 CAMERA 1 2
    """
    lines = readlines_and_prune(file_of_poses)
    print(f"✓ Loaded {len(lines)} poses.")
    poses = [parse_single_pose(line) for line in lines]
    return {d['frame_id']: d for d in poses} # Map frame_id to pose

def make_desired_images_txt(poses: dict[int, dict], id_to_uid: dict[int, int]):
    """
    10 -0.037854 0.723584 -0.127127 0.677372 -0.361500 -0.270088 -0.309526 0 92844212418260.jpg mat4x4(...)
    ...
    """
    lines = []
    for frame_id, uuid in id_to_uid.items():
        pose_desc = poses[frame_id]
        qw, qx, qy, qz = pose_desc['qw'], pose_desc['qx'], pose_desc['qy'], pose_desc['qz']
        tx, ty, tz = pose_desc['tx'], pose_desc['ty'], pose_desc['tz']
        # This is a fake image path, but it is fine to cheat png_folder class.
        lines.append(f"{frame_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 0 {uuid}.png mat4x4(...)")
    return lines

def make_desired_intrinsics_txt(camera_desc: dict):
    """
    id type w h fx fy cx cy 0 0 0 0 0
    0 PINHOLE 480 640 509.454956 509.518982 236.628983 316.687012 0.142826 -0.460981 0.489099 0.000049 -0.000466
    """
    type_ = camera_desc['model']
    w, h = camera_desc['width'], camera_desc['height']
    fx, fy, cx, cy = camera_desc['fx'], camera_desc['fy'], camera_desc['cx'], camera_desc['cy']
    return f"0 {type_} {w} {h} {fx} {fy} {cx} {cy} 0 0 0 0 0"

def load_points3D(file_of_points3D: Path) -> trimesh.PointCloud:
    """
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (FRAME_ID, POINT2_IDX)
    1 -2.5199532488547658 -2.4739193620089974 8.4415616751354854 131 104 88 0.48947664917642042 10
    ...
    """

    # only need x y z r g b
    points3D_xyz = np.loadtxt(file_of_points3D, dtype=np.float32, usecols=(1, 2, 3))
    points3D_rgb = np.loadtxt(file_of_points3D, dtype=np.uint8, usecols=(4, 5, 6))
    return trimesh.PointCloud(vertices=points3D_xyz, colors=points3D_rgb)

def global_scale(points3D: trimesh.PointCloud) -> tuple[np.ndarray, float]:
    """Ensure std = 1"""
    vertices = np.array(points3D.vertices, dtype=np.float64)
    mean = vertices.mean(axis=0)
    std = (vertices - mean).std()
    print(f"✓ Global scale: mean={mean}, std={std}")
    return mean, std if std > 0.01 else 1.0

def _quat_to_rot_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.

    Assumes right-handed coordinate system and unit quaternion.
    """
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        # TODO: Clarify how to handle zero-norm quaternions; dataset should not contain this.
        raise ValueError("❌ Zero-norm quaternion encountered.")
    qw, qx, qy, qz = q / n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=np.float64)
    return R

def _normalize_extrinsics_translations(
    extrinsics: dict[int, dict], mean: np.ndarray, std: float
) -> dict[int, dict]:
    """Normalize world-to-camera translations consistent with point normalization.

    We apply x' = (x - mean) / std to world points while keeping rotation R
    unchanged. For a world->camera transform (R, t) where X_cam = R X_world + t,
    the translation must be updated to t' = (t + R * mean) / std to preserve
    projection consistency.

    Args:
        extrinsics: Mapping frame_id -> pose dict containing 'qw','qx','qy','qz','tx','ty','tz'.
        mean: Global mean of points3D.
        std: Global std of points3D (scalar).

    Returns:
        A new dict with updated translations.
    """
    eps = 1e-8
    if std < eps:
        # TODO: Decide behavior for near-zero std; skip normalization for safety.
        print("⚠️ Std too small; skipping extrinsics normalization.")
        return extrinsics

    mean = np.asarray(mean, dtype=np.float64)
    new_extrinsics: dict[int, dict] = {}
    for frame_id, pose in extrinsics.items():
        R = _quat_to_rot_matrix(pose['qw'], pose['qx'], pose['qy'], pose['qz'])
        t = np.array([pose['tx'], pose['ty'], pose['tz']], dtype=np.float64)
        # Derived from camera center normalization and transform consistency
        t_prime = (t + R @ mean) / float(std)
        p = pose.copy()
        p['tx'], p['ty'], p['tz'] = float(t_prime[0]), float(t_prime[1]), float(t_prime[2])
        new_extrinsics[frame_id] = p
    return new_extrinsics

def main(args):
    print(f"✓ {args.input} -> {args.id} -> {args.output}")

    input_dir = Path(args.input) / args.id
    output_dir = Path(args.output) / args.id
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if not (input_dir / 'images').exists() and not (input_dir / 'images_gt_downsampled').exists():
        raise FileNotFoundError(f"❌ Either {input_dir / 'images'} or {input_dir / 'images_gt_downsampled'} does not exist.")
    if not (input_dir / 'sparse').exists():
        raise FileNotFoundError(f"❌ {input_dir / 'sparse'} does not exist.")
    if not (input_dir / 'train_test_split.json').exists():
        raise FileNotFoundError(f"❌ {input_dir / 'train_test_split.json'} does not exist.")

    with open(input_dir / 'train_test_split.json', 'r') as f:
        data = json.load(f)

    # Where are the input images:
    if (input_dir / 'images').exists():
        image_dir = input_dir / 'images'
    elif (input_dir / 'images_gt_downsampled').exists():
        image_dir = input_dir / 'images_gt_downsampled'
    else:
        raise FileNotFoundError(f"❌ Either {input_dir / 'images'} or {input_dir / 'images_gt_downsampled'} does not exist.")

    shutil.copy(input_dir / 'sparse' / '0' / 'points3D.txt', output_dir / 'points3D.in.txt')
    print(f"✓ Wrote points3D to {output_dir / 'points3D.in.txt'}")
    points3D = load_points3D(output_dir / 'points3D.in.txt')
    if args.normalize:
        mean, std = global_scale(points3D)
        points3D = trimesh.PointCloud(vertices=(np.array(points3D.vertices, dtype=np.float64) - mean) / std, colors=points3D.colors)
    else:
        mean = np.array([0.0, 0.0, 0.0])
        std = 1.0
    points3D.export(output_dir / 'points3D.ply')
    print(f"✓ Wrote points3D to {output_dir / 'points3D.ply'}, {points3D.vertices.shape}, {points3D.colors.shape}")

    (output_dir / 'images').mkdir(parents=True, exist_ok=True)

    # Copy images, with prepend magic number
    train_image_paths: list[str] = data['train'] # List of image paths
    train_images: dict[int, int] = {}
    for relp in train_image_paths:
        in_file_path = Path(image_dir / relp)
        out_file_path = Path(output_dir / 'images' / f'{args.magic_number}{relp}')
        shutil.copy(in_file_path, out_file_path)
        train_images[int(in_file_path.stem)] = int(int(out_file_path.stem)) # e.g. 0000 -> 0 -> 420000
    test_image_paths: list[str] = data['test'] # List of image paths
    test_images: dict[int, int] = {}
    for relp in test_image_paths:
        in_file_path = Path(image_dir / relp)
        out_file_path = Path(output_dir / 'images' / f'{args.magic_number}{relp}')
        shutil.copy(in_file_path, out_file_path)
        test_images[int(in_file_path.stem)] = int(int(out_file_path.stem)) # e.g. 0000 -> 1 -> 420001
    print(f"✓ Loaded {len(train_images)} train images and {len(test_images)} test images.")
    intrinsics = parse_first_camera(input_dir / 'sparse' / '0' / 'cameras.txt')
    extrinsics = parse_poses(input_dir / 'sparse' / '0' / 'frames.txt')

    if args.normalize:
        # Adjust translations consistent with world->camera convention and point normalization.
        # TODO: Confirm frames.txt poses represent world->camera (RIG_FROM_WORLD). If opposite, invert R.
        extrinsics = _normalize_extrinsics_translations(extrinsics, mean, std)

    # build png_folder dataset for them.
    train_desired = make_desired_images_txt(extrinsics, train_images)
    test_desired = make_desired_images_txt(extrinsics, test_images)

    Path(output_dir / 'train_desired.txt').write_text('\n'.join(train_desired))
    Path(output_dir / 'test_desired.txt').write_text('\n'.join(test_desired))
    print(f"✓ Wrote {len(train_desired)} train desired images and {len(test_desired)} test desired images.")

    intrinsics_desired = make_desired_intrinsics_txt(intrinsics)
    Path(output_dir / 'intrinsics.txt').write_text(intrinsics_desired)
    print(f"✓ Wrote intrinsics to {output_dir / 'intrinsics.txt'}")

if __name__ == '__main__':
    main(parse_args())