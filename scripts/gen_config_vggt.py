from pathlib import Path
from argparse import ArgumentParser

PATH_TO_BUILT = Path(__file__).parent / 'build' / 'examples' / 'config_train'
parser = ArgumentParser(description='Train TinyGS')
parser.add_argument('--root', type=str, required=True, help='Path to the scenes, e.g. Final.')
parser.add_argument('--id', type=str, required=True, help='ID of the scene')
parser.add_argument("--working_dir", type=str, default="outputs", help="Path to the working directory.")
parser.add_argument('--out', type=str, required=True, help='Path to the output folder, e.g. out/ARG_ID.')
parser.add_argument("--vggt-time", type=float, required=True, help="VGGT processing time in seconds.")
args = parser.parse_args()

print(f'Launching training for scene {args.id} in {args.root}')

dataroot = Path(args.root)
if not dataroot.exists():
    raise ValueError(f'Path {dataroot} does not exist.')

config_template = r"""
{
  "dataloader": {
    "type": "async",
    "data_type": "float16"
  },
  "dataset": {
    "extrinsics_file_path": "ARG_WORKING_DIR/vggt/images.txt",
    "folder_path": "ARG_WORKING_DIR/images/",
    "intrinsics_file_path": "ARG_WORKING_DIR/vggt/cameras.txt",
    "extension": "png",
    "type": "png_folder"
  },
  "initializer": {
    "default_distance": 0.01,
    "enable_radius_outlier_removal": false,
    "init_opacity": 0.1,
    "init_scaling": 0.6,
    "min_distance": 1.0e-07,
    "nb_points": 16,
    "num_neighbors": 8,
    "radius": 0.05,
    "sh_degree": 3,
    "type": "knn"
  },
  "input_pc_file": "ARG_WORKING_DIR/vggt/init_points.ply",
  "losses": [
    {
      "type": "l1",
      "weight": 0.8
    },
    {
      "type": "fused_ssim",
      "weight": 0.2
    }
  ],
  "lr_scheduler": {
    "decay_rate": 0.9998,
    "initial_lr": 1.0,
    "step_count": 0,
    "type": "exponential"
  },
  "metrics": [
    "psnr"
  ],
  "optimizer": {
    "epsilon": 1.0e-8,
    "max_grad_1": 1.0,
    "means_lr": 0.00016,
    "opacities_l1": 0.01,
    "decouple_decay": false,
    "decay_reduction": "mean",
    "opacities_lr": 0.05,
    "rotations_lr": 0.001,
    "scales_l1": 0.01,
    "scales_lr": 0.005,
    "shs_lr": 0.0025,
    "skip_zero_grad": true,
    "trust_ratio_min": 0.01,
    "trust_ratio_max": 10.0,
    "type": "simple_adam"
  },
  "pose_opt": {
    "type": "adamw",
    "lr": 0.0,
    "momentum": 0.95,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.1
  },
  "rasterizer": {
    "type": "fastgs"
  },
  "strategy": {
    "duplicate_grad_threshold": 0.0008,
    "absgrad": true,
    "duplicate_scale_threshold": 0.005,
    "end_refine": 15000,
    "max_num_gaussians": 3000000,
    "max_screen_size": 10,
    "pruning_opacity_threshold": 0.005,
    "pruning_scale_threshold": 0.1,
    "refine_every": 100,
    "reset_every": 3000,
    "reset_reset_optimizer": false,
    "seed": 42,
    "start_refine": 500,
    "noise_lr_init": 0.0,
    "split_distance": 0.4,
    "opacity_reduction": 0.6,
    "type": "improved"
  },
  "trainer": {
    "accumulate_grad_steps": 1,
    "train_data_type": "float16",
    "eval_data_type": "float16",
    "checkpoint_interval": 1000,
    "early_stopping_patience": 1000,
    "early_stopping_threshold": 1.0e-6,
    "enable_early_stopping": false,
    "far_plane": 100.0,
    "grad_scaler": 128.0,
    "log_interval": 100,
    "max_sh_degree": 3,
    "max_steps": 30000,
    "near_plane": 0.01,
    "sh_degree_interval": 1000,
    "scene_scale_recompute_interval": 1000,
    "reorder_gaussians_interval": 1000,
    "enable_progressive_resolution": false,
    "start_pose_opt": 50000,
    "record_trajectory": false,
    "resolution_milestones": [0, 3000, 5000],
    "resolution_scales": [0.5, 0.75, 1.0],
    "test_steps": [7000, 30000],
    "out_dir": "outputs",
    "export_rasterized": true,
    "max_seconds": ARG_MAX_SECONDS
  }
}
"""

avail_time = 60 - args.vggt_time - 1 # 1 second for other stuff.
print(f'Available time for training: {avail_time:.2f} seconds')

with open(args.out, 'w') as f:
    f.write(
        config_template
        .replace("ARG_DATA", str(dataroot))
        .replace("ARG_ID", args.id)
        .replace("ARG_WORKING_DIR", args.working_dir)
        .replace("ARG_OUT", str(Path(args.out).parent))
        .replace("ARG_MAX_SECONDS", str(avail_time))
    )