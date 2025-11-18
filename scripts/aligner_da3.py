import os
from time import perf_counter
import numpy as np
from camera import (
    CameraExtrinsic,
    CameraIntrinsic,
    calc_w2c,
    load_camera_intrinsics,
    load_camera_extrinsics,
)
from pathlib import Path
from argparse import ArgumentParser

import trimesh
from depth_anything_3.api import DepthAnything3
from PIL import Image

def run_aligner(
    points,
    colors,
    images: list[str],
    camera_extrinsics: list[CameraExtrinsic],
    camera_intrinsics: CameraIntrinsic,
    da3: DepthAnything3,
) -> tuple[np.ndarray, np.ndarray]:
    # depth = da3.inferrence
    w2c = [calc_w2c(extrinsic) for extrinsic in camera_extrinsics]
    w2c = np.array(w2c) # (n, 4, 4)
    intr = np.array([[camera_intrinsics.fx, 0, camera_intrinsics.cx],
                     [0, camera_intrinsics.fy, camera_intrinsics.cy],
                     [0, 0, 1]]) # (3, 3)
    intr = np.tile(intr[None, ...], (w2c.shape[0], 1, 1)) # (n, 3, 3)
    w, h = camera_intrinsics.width, camera_intrinsics.height
    input_images: list[Image.Image] = [Image.open(img).resize((w, h)) for img in images]
    print(f"📕 num input images: {len(input_images)}")
    pred = da3.inference(
        image=input_images,
        extrinsics = w2c,
        intrinsics = intr,
        process_res_method='lower_bound_resize',
        infer_gs=True
    )

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
    print(f"Downscaled camera: w={camera_intrinsics.width}, h={camera_intrinsics.height}")

    # --- Load camera extrinsics ---
    camera_extrinsics, qs, ts = load_camera_extrinsics(EXTRIN_FILE)
    print(f"Parsed {len(qs)} camera poses")
    camera_extrinsics = camera_extrinsics[::args.t_interval]
    qs = qs[::args.t_interval]
    ts = ts[::args.t_interval]

    # --- Load point cloud ---
    pc = trimesh.load(PC_FILE)
    points = np.array(pc.vertices)
    colors = np.array(pc.colors)[:, :3] / 255.0

    print(f"Loaded {points.shape[0]} 3D points")

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
    da3 = da3.to("cuda")
    print("✅ DepthAnything3 model loaded")

    start_time = perf_counter()
    try:
        points, colors = run_aligner(
            points,
            colors,
            images,
            camera_extrinsics,
            camera_intrinsics,
            da3,
        )
        end_time = perf_counter()
        print(f"🚀 Alignment time: {end_time - start_time}")
        trimesh.PointCloud(vertices=points, colors=colors).export(OUT_FILE)
        Path(TIME_FILE).write_text(f"{int(60 - np.round(end_time - start_time))}")
    except Exception as e:
        print(f"❗️ Error: {e}")