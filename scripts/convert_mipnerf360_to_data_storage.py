from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True, slots=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True, slots=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass(frozen=True, slots=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3d_xyz: np.ndarray
    points3d_rgb: np.ndarray


_CAMERA_MODEL_ID_TO_NAME: dict[int, str] = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

_CAMERA_MODEL_PARAM_COUNT: dict[str, int] = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}


def _read_next(fid: BinaryIO, num_bytes: int, fmt: str) -> tuple:
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(f"Unexpected EOF while reading COLMAP binary data: expected {num_bytes} bytes.")
    return struct.unpack("<" + fmt, data)


def read_intrinsics_binary(path: str) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path, "rb") as fid:
        num_cameras = int(_read_next(fid, 8, "Q")[0])
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next(fid, 24, "iiQQ")
            model_name = _CAMERA_MODEL_ID_TO_NAME[int(model_id)]
            num_params = _CAMERA_MODEL_PARAM_COUNT[model_name]
            params = np.asarray(_read_next(fid, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[int(camera_id)] = ColmapCamera(
                camera_id=int(camera_id),
                model=model_name,
                width=int(width),
                height=int(height),
                params=params,
            )
    return cameras


def read_extrinsics_binary(path: str) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with open(path, "rb") as fid:
        num_images = int(_read_next(fid, 8, "Q")[0])
        for _ in range(num_images):
            record = _read_next(fid, 64, "idddddddi")
            image_id = int(record[0])
            qvec = np.asarray(record[1:5], dtype=np.float64)
            tvec = np.asarray(record[5:8], dtype=np.float64)
            camera_id = int(record[8])

            name_bytes = bytearray()
            while True:
                byte = _read_next(fid, 1, "c")[0]
                if byte == b"\x00":
                    break
                name_bytes.extend(byte)
            name = name_bytes.decode("utf-8")

            num_points2d = int(_read_next(fid, 8, "Q")[0])
            # Skip x, y, point3d_id triplets. We do not use this data for training.
            fid.seek(24 * num_points2d, 1)
            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


def read_points3d_binary(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fid:
        num_points = int(_read_next(fid, 8, "Q")[0])
        xyzs = np.empty((num_points, 3), dtype=np.float64)
        rgbs = np.empty((num_points, 3), dtype=np.uint8)
        for point_idx in range(num_points):
            record = _read_next(fid, 43, "QdddBBBd")
            xyzs[point_idx, :] = np.asarray(record[1:4], dtype=np.float64)
            rgbs[point_idx, :] = np.asarray(record[4:7], dtype=np.uint8)
            track_length = int(_read_next(fid, 8, "Q")[0])
            # Skip track items (image_id, point2d_idx).
            fid.seek(8 * track_length, 1)
    return xyzs, rgbs


def read_intrinsics_text(path: str) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path, "r", encoding="utf-8") as fid:
        for line in fid:
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.asarray(tuple(float(v) for v in parts[4:]), dtype=np.float64)
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_extrinsics_text(path: str) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with open(path, "r", encoding="utf-8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split()
            image_id = int(parts[0])
            qvec = np.asarray(tuple(float(v) for v in parts[1:5]), dtype=np.float64)
            tvec = np.asarray(tuple(float(v) for v in parts[5:8]), dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]

            # Read and discard the 2D points line.
            _ = fid.readline()
            images[image_id] = ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


def read_points3d_text(path: str) -> tuple[np.ndarray, np.ndarray]:
    xyzs: list[list[float]] = []
    rgbs: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as fid:
        for line in fid:
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            parts = row.split()
            xyzs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgbs.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not xyzs:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
    return np.asarray(xyzs, dtype=np.float64), np.asarray(rgbs, dtype=np.uint8)


def load_colmap_model(colmap_sparse_root: str) -> ColmapModel:
    """Load COLMAP cameras/images/points from either binary or text files."""
    cameras_bin = f"{colmap_sparse_root}/cameras.bin"
    images_bin = f"{colmap_sparse_root}/images.bin"
    points_bin = f"{colmap_sparse_root}/points3D.bin"
    cameras_txt = f"{colmap_sparse_root}/cameras.txt"
    images_txt = f"{colmap_sparse_root}/images.txt"
    points_txt = f"{colmap_sparse_root}/points3D.txt"

    try:
        cameras = read_intrinsics_binary(cameras_bin)
        images = read_extrinsics_binary(images_bin)
        points_xyz, points_rgb = read_points3d_binary(points_bin)
    except FileNotFoundError:
        cameras = read_intrinsics_text(cameras_txt)
        images = read_extrinsics_text(images_txt)
        points_xyz, points_rgb = read_points3d_text(points_txt)

    return ColmapModel(
        cameras=cameras,
        images=images,
        points3d_xyz=points_xyz,
        points3d_rgb=points_rgb,
    )


def write_point_cloud_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write an RGB point cloud as ASCII PLY.

    ASCII is intentional here: this file is a dataset exchange artifact used for
    debugging and inspection, not a runtime-critical path.
    """

    xyz_array = np.asarray(xyz, dtype=np.float32)
    rgb_array = np.asarray(rgb, dtype=np.float32)
    if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N,3], got {tuple(xyz_array.shape)}.")
    if rgb_array.shape != xyz_array.shape:
        raise ValueError(
            f"rgb must match xyz shape [N,3], got {tuple(rgb_array.shape)} vs {tuple(xyz_array.shape)}."
        )

    rgb_uint8 = np.clip(np.round(rgb_array * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz_array.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(output_path, "w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(header_lines))
        output.write("\n")
        for idx in range(xyz_array.shape[0]):
            x, y, z = xyz_array[idx]
            r, g, b = rgb_uint8[idx]
            output.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


# --- Original Script Logic ---


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Convert a MipNeRF360/COLMAP scene into the dataset layout "
            "described in docs/data.md."
        )
    )
    parser.add_argument("--source_path", type=str, required=True, help="COLMAP scene root (contains sparse/0).")
    parser.add_argument("--output_path", type=str, required=True, help="Output dataset root path.")
    parser.add_argument("--images", type=str, default="images", help="Image subfolder under source_path.")
    parser.add_argument("--eval", action="store_true", default=False, help="Create val split via LLFF holdout.")
    parser.add_argument("--llffhold", type=int, default=8, help="Holdout period when --eval is enabled.")
    parser.add_argument(
        "--image_mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="How to place images in output splits.",
    )
    return parser


def _extract_camera_record(camera: ColmapCamera) -> dict[str, object]:
    if camera.model == "SIMPLE_PINHOLE":
        params = [float(camera.params[0]), float(camera.params[1]), float(camera.params[2])]
    elif camera.model == "PINHOLE":
        params = [
            float(camera.params[0]),
            float(camera.params[1]),
            float(camera.params[2]),
            float(camera.params[3]),
        ]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model {camera.model!r} in conversion. "
            "Supported values are SIMPLE_PINHOLE and PINHOLE."
        )

    return {
        "camera_id": int(camera.camera_id),
        "model": str(camera.model),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": params,
    }


def _extract_pose_record(image: ColmapImage) -> dict[str, object]:
    return {
        "image_id": int(image.image_id),
        "camera_id": int(image.camera_id),
        "name": str(image.name),
        "qvec": [float(value) for value in image.qvec.tolist()],
        "tvec": [float(value) for value in image.tvec.tolist()],
    }


def _copy_or_link_image(source: Path, target: Path, *, image_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()

    if image_mode == "copy":
        shutil.copy2(source, target)
    elif image_mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported image_mode {image_mode!r}.")


def _write_split(
    split_root: Path,
    selected_images: list[ColmapImage],
    cameras_by_id: dict[int, ColmapCamera],
    *,
    images_root: Path,
    image_mode: str,
) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    images_output_root = split_root / "images"
    images_output_root.mkdir(parents=True, exist_ok=True)

    used_camera_ids = sorted({int(image.camera_id) for image in selected_images})
    camera_records = [_extract_camera_record(cameras_by_id[camera_id]) for camera_id in used_camera_ids]
    pose_records = [_extract_pose_record(image) for image in selected_images]

    for image in selected_images:
        source_image = images_root / image.name
        if not source_image.exists():
            raise FileNotFoundError(f"Image referenced by COLMAP does not exist: {source_image}")
        output_image = images_output_root / image.name
        _copy_or_link_image(source_image, output_image, image_mode=image_mode)

    with open(split_root / "cameras.json", "w", encoding="utf-8") as output:
        json.dump(camera_records, output, indent=2, sort_keys=True)
    with open(split_root / "poses.json", "w", encoding="utf-8") as output:
        json.dump(pose_records, output, indent=2, sort_keys=True)


def convert_colmap_scene_to_dataset_storage(
    *,
    source_path: str,
    output_path: str,
    images_dir: str,
    eval_mode: bool,
    llffhold: int,
    image_mode: str,
) -> None:
    source_root = Path(source_path)
    sparse_root = source_root / "sparse" / "0"
    if not sparse_root.exists():
        raise FileNotFoundError(f"COLMAP sparse directory was not found: {sparse_root}")

    images_root = source_root / images_dir
    if not images_root.exists():
        raise FileNotFoundError(f"Image directory was not found: {images_root}")

    model = load_colmap_model(str(sparse_root))
    all_images_sorted = sorted(model.images.values(), key=lambda image: image.name)
    if not all_images_sorted:
        raise RuntimeError(f"No registered images found in COLMAP model under {sparse_root}.")

    hold = max(1, int(llffhold))
    val_name_set: set[str]
    if bool(eval_mode):
        val_name_set = {image.name for index, image in enumerate(all_images_sorted) if index % hold == 0}
    else:
        val_name_set = set()

    train_images = [image for image in all_images_sorted if image.name not in val_name_set]
    val_images = [image for image in all_images_sorted if image.name in val_name_set]
    if not train_images:
        raise RuntimeError("No training images remain after split. Adjust --llffhold or disable --eval.")

    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_split(
        output_root / "train",
        train_images,
        model.cameras,
        images_root=images_root,
        image_mode=image_mode,
    )
    if bool(eval_mode):
        _write_split(
            output_root / "val",
            val_images,
            model.cameras,
            images_root=images_root,
            image_mode=image_mode,
        )

    points_xyz = np.asarray(model.points3d_xyz, dtype=np.float32)
    points_rgb = np.asarray(model.points3d_rgb, dtype=np.float32)
    if int(points_xyz.shape[0]) > 0:
        write_point_cloud_ply(
            str(output_root / "train" / "points3d.ply"),
            points_xyz,
            np.clip(points_rgb / 255.0, 0.0, 1.0),
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    convert_colmap_scene_to_dataset_storage(
        source_path=args.source_path,
        output_path=args.output_path,
        images_dir=args.images,
        eval_mode=bool(args.eval),
        llffhold=int(args.llffhold),
        image_mode=str(args.image_mode),
    )


if __name__ == "__main__":
    main()
