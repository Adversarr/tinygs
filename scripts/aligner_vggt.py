import os
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from time import time
from argparse import ArgumentParser


# Ensure project root (scripts/) is in sys.path so `vggt.*` imports work
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.eval_utils import load_images_rgb, get_vgg_input_imgs


def write_extrin_intrin_file(
    extrin_matrix: np.ndarray,
    intrins: np.ndarray,
    paths: list[Path],
    export_dir: Path, H: int, W: int
):
    extr = []
    intr = []

    # Line of Extr:
    # 30 -0.059927 0.586125 -0.085150 0.803502 0.436363 0.285773 0.343262 0 268086079118000.jpg mat4x4((-0.305732, 0.196121, -0.931700, 0.000000), (-0.003514, 0.978316, 0.207087, 0.000000), (0.952111, 0.066587, -0.298414, 0.000000), (0.436363, -0.285773, -0.343262, 1.000000))
    # fid qr qx qy qz x y z cid id mat4x4(...)
    # Line of Intr:
    # 0 PINHOLE 480 640 453.263031 454.014008 239.595001 323.046997 0.107172 -0.259888 0.208784 -0.000184 0.000525
    # cid type width height fx fy cx cy k1 k2 k3 p1 p2

    # em: (3, 4), im: (3, 3)
    for i, (em, im, path) in enumerate(zip(extrin_matrix, intrins, paths)):
        R, t = em[:3, :3], em[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        tx, ty, tz = t
        fx, fy, cx, cy = im[0, 0], im[1, 1], im[0, 2], im[1, 2]
        cid = i
        fid = (1 + i) * 10
        id = path.stem
        ext_line = f"{fid} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cid} {id}.jpg mat4x4(...)"
        int_line = f"{cid} PINHOLE {W} {H} {fx} {fy} {cx} {cy} 0.0 0.0 0.0 0.0 0.0"
        extr.append(ext_line)
        intr.append(int_line)
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / 'images.txt').write_text('\n'.join(extr))
    (export_dir / 'cameras.txt').write_text('\n'.join(intr))

def run_vggt(model: VGGT, vgg_input: torch.Tensor, dtype: torch.dtype, image_paths=None):
    """
    Run VGGT to predict extrinsics, intrinsics, depth map and depth confidence.
    vgg_input: tensor [N, 3, H, W] in [0,1]
    Returns: (extrinsic [N,3,4], intrinsic [N,3,3], depth [N,H,W], depth_conf [N,H,W])
    """
    assert len(vgg_input.shape) == 4 and vgg_input.shape[1] == 3

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            vgg_input_cuda = vgg_input.cuda().to(torch.bfloat16)
            predictions = model(vgg_input_cuda, image_paths=image_paths)

    # Extract predictions
    pose_enc = predictions["pose_enc"]
    H, W = vgg_input.shape[2], vgg_input.shape[3]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

    depth_tensor = predictions["depth"].detach().float().cpu()
    depth_conf_tensor = predictions["depth_conf"].detach().float().cpu()

    # Move to numpy
    depth_np = depth_tensor.numpy()
    depth_conf_np = depth_conf_tensor.numpy()
    extrinsic_np = extrinsic.detach().float().cpu().numpy()
    intrinsic_np = intrinsic.detach().float().cpu().numpy()

    return extrinsic_np[0], intrinsic_np[0], depth_np[0], depth_conf_np[0], pose_enc[0].detach().float().cpu().numpy()

def main():
    parser = ArgumentParser(description="Generate point cloud from images using VGGT (no SLAM aligner)")
    parser.add_argument("--root", type=str, default='/data/yzr/Final', help="Root directory of all scenes")
    parser.add_argument("--id", type=str, default='1750383597053', help="ID of the scene")
    parser.add_argument("--working_dir", type=str, default='output/1750383597053', help="Working directory containing images/")
    parser.add_argument("--ckpt_path", type=str, default='model_tracker_fixed_e30.pt', help="VGGT model checkpoint path")
    parser.add_argument("--merging", type=int, default=0, help="VGGT merging parameter")
    parser.add_argument("--depth_conf_thresh", type=float, default=1, help="Depth confidence threshold")
    parser.add_argument("--max_points", type=int, default=100000, help="Max number of 3D points to keep")
    parser.add_argument("--avg-intr", action='store_true')
    args = parser.parse_args()

    ID = args.id
    INPUT_FOLDER = f"{args.working_dir}/images/"
    OUT_DIR = Path(f"{args.working_dir}/vggt/")
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    OUT_FILE = OUT_DIR / "init_points.ply"
    INPUT_FOLDER = f"{args.working_dir}/images/"

    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*")))
    image_paths = [Path(i) for i in image_paths]
    if len(image_paths) == 0:
        raise ValueError(f"Error: no images found in {INPUT_FOLDER}")
    base_image_names = [os.path.basename(p) for p in image_paths]
    print(f"Loaded {len(image_paths)} images from {INPUT_FOLDER}", file=sys.stderr)

    # Load images as RGB array list and build VGGT input
    original_images = load_images_rgb(image_paths)
    image_height, image_width = original_images[0].shape[0:2]
    print(f"Original height, width={image_height}, {image_width}", file=sys.stderr)
    images = []
    for img in original_images:
        # We are dealing with a portrait image, rotate it to landscape
        images.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)

    # Initialize VGGT
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"Loading VGGT from {args.ckpt_path}", file=sys.stderr)
    model = VGGT(merging=args.merging, vis_attn_map=False)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval().to(dtype)
    model.update_patch_dimensions(patch_width, patch_height)
    print(f"VGGT initialized with patch dimensions: {patch_width}x{patch_height}", file=sys.stderr)

    # Run VGGT
    print("=" * 50, file=sys.stderr)
    print(f"START:: Running VGGT on {len(vgg_input)} images", file=sys.stderr)
    start = time()
    extrinsic, intrinsic, depth_map, depth_conf, pose_enc = run_vggt(model, vgg_input, dtype, base_image_names)
    if args.avg_intr:
        intrisic_mean = np.mean(intrinsic, axis=0)
        intrinsic = np.tile(intrisic_mean[None, ...], (len(vgg_input), 1, 1))

    # Rotate everything back to match original portrait orientation
    # - Inputs were rotated 90° clockwise before VGGT; revert outputs by 90° CCW.
    # - Adjust intrinsics to the rotated-back dimensions (swap fx/fy, reset principal point).
    # - Rotate extrinsics to match camera axes after undoing image rotation.
    depth_map = np.rot90(depth_map, k=1, axes=(1, 2))
    depth_conf = np.rot90(depth_conf, k=1, axes=(1, 2))

    # Rotate vgg_input back so color sampling aligns with depth/points orientation
    vgg_input = torch.rot90(vgg_input, k=1, dims=(2, 3))

    # Update intrinsics: swap fx/fy and center principal point to new width/height
    H_back = depth_map.shape[1]
    W_back = depth_map.shape[2]
    try:
        K_back = intrinsic.copy()
        if K_back.ndim == 2:
            # Single camera case: shape [3,3]
            fx_rot, fy_rot = K_back[0, 0], K_back[1, 1]
            K_back[0, 0] = fy_rot
            K_back[1, 1] = fx_rot
            K_back[0, 2] = W_back / 2.0
            K_back[1, 2] = H_back / 2.0
            K_back[2, 2] = 1.0
        else:
            # Batched cameras: shape [S,3,3]
            fx_rot = K_back[:, 0, 0].copy()
            fy_rot = K_back[:, 1, 1].copy()
            K_back[:, 0, 0] = fy_rot
            K_back[:, 1, 1] = fx_rot
            K_back[:, 0, 2] = W_back / 2.0
            K_back[:, 1, 2] = H_back / 2.0
            K_back[:, 2, 2] = 1.0
        intrinsic = K_back
    except Exception as e:
        # TODO: Intrinsics rotation failed; investigate shape/type. Using original intrinsics.
        # Reason: unexpected intrinsics shape or dtype caused exception: {e}
        print(e, file=sys.stderr)
        pass

    # Rotate extrinsics (world->cam [R|t]) to align with rotated-back camera axes.
    # S_ccw is +90° about z-axis in OpenCV camera coords (x-right, y-down, z-forward).
    # New extrinsics: R' = S_ccw @ R, t' = S_ccw @ t
    S_ccw = Rotation.from_euler('z', -90, degrees=True).as_matrix().astype(np.float32)
    try:
        if extrinsic.ndim == 3 and extrinsic.shape[1:] == (3, 4):
            extrinsic[:, :, :3] = S_ccw[None] @ extrinsic[:, :, :3]
            extrinsic[:, :, 3] = (S_ccw @ extrinsic[:, :, 3].T).T
        elif extrinsic.ndim == 2 and extrinsic.shape == (3, 4):
            extrinsic[:, :3] = S_ccw @ extrinsic[:, :3]
            extrinsic[:, 3] = S_ccw @ extrinsic[:, 3]
        else:
            # TODO: Unexpected extrinsic shape; skip rotation.
            # Reason: shape not in {(3,4), (S,3,4)}
            pass
    except Exception as e:
        # TODO: Extrinsics rotation failed; investigate numeric stability / dtype.
        # Reason: exception during S_ccw application: {e}
        print(e, file=sys.stderr)

    print(f"Extrinsic shape: {extrinsic.shape}", file=sys.stderr)
    print(f"Intrinsic shape: {intrinsic.shape}", file=sys.stderr)
    print(f"Depth map shape: {depth_map.shape}", file=sys.stderr)
    print(f"Depth conf shape: {depth_conf.shape}", file=sys.stderr)
    print(f"Pose enc shape: {pose_enc.shape}", file=sys.stderr)

    # Back-project depth to 3D points (world coords)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Prepare RGB colors aligned to point grid resolution (518x294 in feedforward mode)
    vggt_fixed_resolution_height = depth_map.shape[1]
    vggt_fixed_resolution_width = depth_map.shape[2]
    points_rgb = F.interpolate(
        vgg_input,
        size=(vggt_fixed_resolution_height, vggt_fixed_resolution_width),
        mode="bilinear",
        align_corners=False,
    )
    points_rgb = (points_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)  # [N,H,W,3]

    # Build frame-pixel grid and filter by confidence
    num_frames, height, width, _ = points_3d.shape
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    conf_mask = depth_conf >= args.depth_conf_thresh
    conf_mask = randomly_limit_trues(conf_mask, args.max_points)

    points_3d = points_3d[conf_mask]
    print(f"Filtered {points_3d.shape[0]} points by confidence", file=sys.stderr)
    points_rgb = points_rgb[conf_mask]
    points_xyf = points_xyf[conf_mask]  # not used for PLY but kept for completeness
    write_extrin_intrin_file(
        extrinsic,
        intrinsic,
        image_paths,
        OUT_DIR,
        vggt_fixed_resolution_height, vggt_fixed_resolution_width
    )
    end_time = time()
    print(f"[INFO] VGGT processing time: {end_time - start:.4f} seconds")
    print(f"[INFO] VGGT processing time: {end_time - start:.4f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # Save PLY point cloud
    try:
        import trimesh
        from pcloudsim import simplify_point_cloud, RemovalParams
        mean = points_3d.mean(axis=0)
        scale = np.abs(points_3d - mean).max()
        params = RemovalParams(
            enable_statistical_outliers=True,
            n_neighbors_stats=20,
            std_dev_mul=2.0,
            enable_radius_outliers=True,
            radius=0.05 * scale,
            min_points_radius=16,
            enable_voxel_simplify=False,
            voxel_size=0.001 * scale
        )
        points_3d, points_rgb = simplify_point_cloud(points_3d, points_rgb, params)
        pc = trimesh.PointCloud(points_3d, colors=points_rgb.astype(np.uint8))
        pc.export(str(OUT_FILE))
        print(f"Exported {points_3d.shape[0]} points to {OUT_FILE}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to save PLY: {e}", file=sys.stderr)


if __name__ == "__main__":
    with torch.no_grad():
        main()