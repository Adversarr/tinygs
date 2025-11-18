import os
from time import perf_counter
import numpy as np
from camera import (
    CameraExtrinsic,
    CameraIntrinsic,
    calc_w2c,
    calc_c2w,
    load_camera_intrinsics,
    load_camera_extrinsics,
)
from pathlib import Path
from argparse import ArgumentParser

import trimesh
from depth_anything_3.api import DepthAnything3, align_poses_umeyama
from depth_anything_3.utils.export.gs import save_gaussian_ply
from PIL import Image
import torch

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

@torch.no_grad()
def run_aligner(
    images: list[str],
    camera_extrinsics: list[CameraExtrinsic],
    camera_intrinsics: CameraIntrinsic,
    da3: DepthAnything3,
    out_file: str,
) -> tuple[np.ndarray, np.ndarray]:
    # depth = da3.inferrence
    w2c = [calc_w2c(extrinsic) for extrinsic in camera_extrinsics]
    w2c = np.array(w2c) # (n, 4, 4)
    intr = np.array([[camera_intrinsics.fx, 0, camera_intrinsics.cx],
                     [0, camera_intrinsics.fy, camera_intrinsics.cy],
                     [0, 0, 1]]) # (3, 3)
    intr = np.tile(intr[None, ...], (w2c.shape[0], 1, 1)) # (n, 3, 3)
    w, h = camera_intrinsics.width, camera_intrinsics.height
    print(f"📕 num input images: {len(images)}")
    pred = da3.inference(
        image=[Image.open(img).resize((w, h)) for img in images],
        extrinsics = w2c,
        intrinsics = intr,
        process_res_method='upper_bound_resize',
        infer_gs=True,
    )
    gs = pred.gaussians
    assert gs is not None
    s = pred.alignment_scale
    t = pred.alignment_translation
    r = pred.alignment_rotation
    # ✓ scale: 0.1305, t: [ 0.03388288 -0.14224175  0.39337836], r: [[ 0.9747912  -0.12696111  0.18347477]...
    # r, t, s = align_poses_umeyama(pred.extrinsics, w2c[:, :3, :])
    print(f"✓ scale: {s:.4f}, t: {t}, r: {r}")

    points = gs.means.detach().cpu().numpy()[0] # (n, 3)
    points = (points - t.reshape(-1, 3) * (1 / s - 1))
    # points = (points * scale) @ r.T + t.reshape(-1, 3)
    colors = gs.harmonics.detach().cpu().numpy()[0, ..., 0] # (n, 3)
    opacities = gs.opacities.detach().cpu().numpy()[0] # (n,)
    mask = opacities > 0.0
    if pred.conf is not None:
        conf = pred.conf
        print(f"Use confidence map to mask points with confidence > 0.5 percentile")
        c_percentile = np.percentile(conf, 50, axis=(1, 2))
        print(c_percentile.shape)
        print(f"✓ Conf Threshold: {c_percentile.mean():.4f}")
        mask &= (conf > c_percentile[:, None, None]).flatten()

    # Boundaries
    n, h, w = pred.depth.shape
    gstrim_h = int(8 / 256 * h)
    gstrim_w = int(8 / 256 * w)
    b_mask = np.zeros((n, h, w), dtype=bool)
    b_mask[:, gstrim_h:-gstrim_h, gstrim_w:-gstrim_w] = 1
    mask &= b_mask.flatten()

    ctx_depth = pred.depth
    d_percentile = np.percentile(ctx_depth, 80, axis=(1, 2))
    print(f"✓ Depth Threshold: {d_percentile.mean():.4f}")
    mask &= (ctx_depth < d_percentile[:, None, None]).flatten()

    print(f"📕 num predicted points: {points.shape}, colors: {colors.shape}, opacities: {opacities.shape}, valid: {mask.sum()}")
    points = points[mask]
    colors = colors[mask]

    # # # Align the cameras.
    # T_colmap = w2c[:, :3, :] # (n, 3, 4)
    # T_pred = pred.extrinsics # (n, 3, 4)
    # r, t, s = align_poses_umeyama(T_pred, T_colmap, return_aligned=False)
    # # Apply r, t, s to points
    # print(f"✓ r, t, s = {r}, {t}, {s}")
    # points = (points * s) @ r.T + t.reshape(-1, 3)
    return points, (colors * 0.2820948 + 0.5).clip(0, 1) # sh to 01

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--working_dir", type=str, help="Working directory", default='out2/1747834320424')
    parser.add_argument("--downsampling", type=int, help="Downsampling factor", default=504)
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
    print(f"Downscaled camera by factor {downscale:.2e}: w={camera_intrinsics.width}, h={camera_intrinsics.height}")

    # --- Load camera extrinsics ---
    camera_extrinsics, qs, ts = load_camera_extrinsics(EXTRIN_FILE)
    print(f"Parsed {len(qs)} camera poses")
    camera_extrinsics = camera_extrinsics[::args.t_interval]
    qs = qs[::args.t_interval]
    ts = ts[::args.t_interval]

    # --- Load point cloud ---
    pc = trimesh.load(PC_FILE)
    init_points = np.array(pc.vertices)
    init_colors = np.array(pc.colors)[:, :3] / 255.0

    print(f"Loaded {init_points.shape[0]} 3D points")

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

    print("💾 Loading DepthAnything3 model...")
    da3 = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
    da3 = da3.to("cuda").eval()
    print("✅ DepthAnything3 model loaded")

    start_time = perf_counter()
    try:
        points, colors = run_aligner(
            images,
            camera_extrinsics,
            camera_intrinsics,
            da3,
            OUT_FILE
        )
        end_time = perf_counter()
        print(f"🚀 Alignment time: {end_time - start_time}")

        from pcloudsim import simplify_point_cloud, RemovalParams
        params = RemovalParams(
            enable_statistical_outliers=True,
            std_dev_mul=3,
            enable_radius_outliers=True,
            radius=0.1,
            enable_voxel_simplify=False,
            voxel_size=0.005,
        )
        points, colors = simplify_point_cloud(points, colors, params)

        # current_sparse_pc, final_transform, prev_mse = align_sparse_to_dense(init_points, points, max_iterations=10)
        # points = np.concatenate([current_sparse_pc, points])
        # colors = np.concatenate([init_colors, colors])

        points = np.concatenate([init_points, points])
        colors = np.concatenate([init_colors, colors])

        trimesh.PointCloud(vertices=points, colors=colors).export(OUT_FILE)
        Path(TIME_FILE).write_text(f"{int(60 - np.round(end_time - start_time))}")
    except Exception as e:
        print(f"❗️ Error: {e}")
        raise