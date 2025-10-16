import os
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# Ensure project root (scripts/) is in sys.path so `vggt.*` imports work
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from argparse import ArgumentParser

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.eval_utils import load_images_rgb, get_vgg_input_imgs


def run_vggt(model: VGGT, vgg_input: torch.Tensor, dtype: torch.dtype, image_paths=None):
    """
    Run VGGT to predict extrinsics, intrinsics, depth map and depth confidence.
    vgg_input: tensor [N, 3, H, W] in [0,1]
    Returns: (extrinsic [N,4,4], intrinsic [N,3,3], depth [N,H,W], depth_conf [N,H,W])
    """
    assert len(vgg_input.shape) == 4 and vgg_input.shape[1] == 3

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            vgg_input_cuda = vgg_input.cuda().to(torch.bfloat16)
            predictions = model(vgg_input_cuda, image_paths=image_paths)

    # Extract predictions
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
    )

    depth_tensor = predictions["depth"].detach().float().cpu()
    depth_conf_tensor = predictions["depth_conf"].detach().float().cpu()

    # Move to numpy
    depth_np = depth_tensor.numpy()
    depth_conf_np = depth_conf_tensor.numpy()
    extrinsic_np = extrinsic.detach().float().cpu().numpy()
    intrinsic_np = intrinsic.detach().float().cpu().numpy()

    return extrinsic_np[0], intrinsic_np[0], depth_np[0], depth_conf_np[0]


def main():
    parser = ArgumentParser(description="Generate point cloud from images using VGGT (no SLAM aligner)")
    parser.add_argument("--root", type=str, default='/data/yzr/Final', help="Root directory of all scenes")
    parser.add_argument("--id", type=str, default='1750383597053', help="ID of the scene")
    parser.add_argument("--working_dir", type=str, default='output/1750383597053', help="Working directory containing images/")
    parser.add_argument("--out", type=str, default='aligned_points/', help="Output directory for generated PLY")
    parser.add_argument("--ckpt_path", type=str, default='model_tracker_fixed_e30.pt', help="VGGT model checkpoint path")
    parser.add_argument("--merging", type=int, default=0, help="VGGT merging parameter")
    parser.add_argument("--depth_conf_thresh", type=float, default=2, help="Depth confidence threshold")
    parser.add_argument("--max_points", type=int, default=100000, help="Max number of 3D points to keep")
    parser.add_argument("--full-video", action="store_true", help="Process full video instead of extracted frames")
    parser.add_argument("--t_interval", type=int, default=1, help="Time interval between frames to process")
    args = parser.parse_args()

    ID = args.id
    INPUT_FOLDER = f"{args.working_dir}/images/"
    VIDEO_FILE = f'{args.root}/{ID}/{ID}_flip.mp4'
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    OUT_FILE = OUT_DIR / f"{ID}.ply"

    # Gather images
    if args.full_video:
        print(f"Extracting frames from {VIDEO_FILE}")
        cap = cv2.VideoCapture(VIDEO_FILE)
        Path(f"{args.working_dir}/full_frames").mkdir(exist_ok=True, parents=True)
        frame_count = 0
        max_resolution_wh = 518
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame if larger than max_resolution_wh
            if max(frame.shape[:2]) > max_resolution_wh:
                scale = max_resolution_wh / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(f"{args.working_dir}/full_frames/{frame_count:06d}.png", frame)
            frame_count += 1
        cap.release()
        INPUT_FOLDER = f"{args.working_dir}/full_frames/"
    else:
        INPUT_FOLDER = f"{args.working_dir}/images/"
    image_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*")))
    # Filter image paths based on t_interval
    image_paths = image_paths[::args.t_interval]
    
    if len(image_paths) == 0:
        print(f"Error: no images found in {INPUT_FOLDER}")
        return
    base_image_names = [os.path.basename(p) for p in image_paths]
    print(f"Loaded {len(image_paths)} images from {INPUT_FOLDER}")

    # Load images as RGB array list and build VGGT input
    images = load_images_rgb(image_paths)
    images_array = np.stack(images)
    vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)

    # Initialize VGGT
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"Loading VGGT from {args.ckpt_path}")
    model = VGGT(merging=args.merging, vis_attn_map=False)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval().to(torch.bfloat16)
    model.update_patch_dimensions(patch_width, patch_height)
    print(f"VGGT initialized with patch dimensions: {patch_width}x{patch_height}")


    # Run VGGT
    print(f"Running VGGT on {len(vgg_input)} images")
    extrinsic, intrinsic, depth_map, depth_conf = run_vggt(model, vgg_input, dtype, base_image_names)

    print(f"Extrinsic shape: {extrinsic.shape}")
    print(f"Intrinsic shape: {intrinsic.shape}")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth conf shape: {depth_conf.shape}")

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
    print(f"Filtered {points_3d.shape[0]} points by confidence")
    points_rgb = points_rgb[conf_mask]
    points_xyf = points_xyf[conf_mask]  # not used for PLY but kept for completeness
    points_3d[:, 1] *= -1

    # Save PLY point cloud
    try:
        import trimesh
        pc = trimesh.PointCloud(points_3d, colors=points_rgb)
        pc.export(str(OUT_FILE))
        print(f"Exported {points_3d.shape[0]} points to {OUT_FILE}")
    except Exception as e:
        print(f"Failed to save PLY: {e}")


if __name__ == "__main__":
    with torch.no_grad():
        main()