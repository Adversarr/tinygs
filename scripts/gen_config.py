from subprocess import run
from pathlib import Path
from argparse import ArgumentParser

PATH_TO_BUILT = Path(__file__).parent / 'build' / 'examples' / 'config_train'
parser = ArgumentParser(description='Train TinyGS')
parser.add_argument('--root', type=str, required=True, help='Path to the scenes, e.g. Final.')
parser.add_argument('--id', type=str, required=True, help='ID of the scene')
parser.add_argument("--working_dir", type=str, default="outputs", help="Path to the working directory.")
parser.add_argument('--out', type=str, required=True, help='Path to the output folder, e.g. out/ARG_ID.')
args = parser.parse_args()

print(f'Launching training for scene {args.id} in {args.root}')

dataroot = Path(args.root)
if not dataroot.exists():
    raise ValueError(f'Path {dataroot} does not exist.')

config_template = r"""
{
  "dataloader": {
    "type": "async"
  },
  "dataset": {
    "extrinsics_file_path": "ARG_DATA/ARG_ID/inputs/slam/images.txt",
    "folder_path": "ARG_WORKING_DIR/images/",
    "intrinsics_file_path": "ARG_DATA/ARG_ID/inputs/slam/cameras.txt",
    "extension": "png",
    "type": "png_folder"
  },
  "initializer": {
    "default_distance": 0.001,
    "enable_radius_outlier_removal": false,
    "init_opacity": 0.1,
    "init_scaling": 1.0,
    "min_distance": 1.0e-07,
    "nb_points": 16,
    "num_neighbors": 3,
    "radius": 0.05,
    "sh_degree": 3,
    "type": "knn"
  },
  "input_pc_file": "ARG_WORKING_DIR/aligned_points/ARG_ID.ply",
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
    "decay_rate": 0.9996,
    "initial_lr": 1.0,
    "step_count": 0,
    "type": "exponential"
  },
  "metrics": [
    "psnr"
  ],
  "optimizer": {
    "beta1": 0.9,
    "beta2": 0.999,
    "enable_adabound": true,
    "gamma": 1e-3,
    "epsilon": 1.0e-8,
    "max_grad_1": 0.0,
    "means_lr": 0.00016,
    "opacities_l1": 0.01,
    "decouple_decay": true,
    "opacities_lr": 0.05,
    "rotations_lr": 0.001,
    "scales_l1": 0.01,
    "scales_lr": 0.005,
    "shs_lr": 0.0025,
    "skip_zero_grad": false,
    "type": "simple_adam"
  },
  "rasterizer": {
    "type": "fastgs"
  },
  "strategy": {
    "duplicate_grad_threshold": 0.0002,
    "duplicate_scale_threshold": 0.005,
    "end_refine": 25000,
    "max_num_gaussians": 1500000,
    "max_screen_size": 20,
    "pruning_opacity_threshold": 0.005,
    "pruning_scale_threshold": 0.1,
    "refine_every": 100,
    "reset_every": 0,
    "seed": 42,
    "start_refine": 500,
    "noise_lr_init": 80.0,
    "type": "default"
  },
  "trainer": {
    "checkpoint_interval": 1000,
    "early_stopping_patience": 1000,
    "early_stopping_threshold": 1.0e-6,
    "enable_early_stopping": false,
    "far_plane": 100.0,
    "grad_scaler": 10.0,
    "log_interval": 100,
    "max_sh_degree": 3,
    "max_steps": 7001,
    "near_plane": 0.01,
    "sh_degree_interval": 1500,
    "test_steps": [3000, 7000, 30000],
    "out_dir": "ARG_OUT",
    "export_rasterized": true
  }
}
"""

with open(args.out, 'w') as f:
    f.write(
        config_template
        .replace("ARG_DATA", str(dataroot))
        .replace("ARG_ID", args.id)
        .replace("ARG_WORKING_DIR", args.working_dir)
        .replace("ARG_OUT", str(Path(args.out).parent))
    )