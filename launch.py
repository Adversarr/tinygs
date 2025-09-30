from subprocess import run
from pathlib import Path
from argparse import ArgumentParser

PATH_TO_BUILT = Path(__file__).parent / 'build' / 'examples' / 'config_train'
parser = ArgumentParser(description='Train TinyGS')
parser.add_argument('--data', type=str, required=True, help='Path to the scene.')
parser.add_argument('--id', type=str, required=False, help='ID of the scene, infer from `data` if not provided.')


config_template = """
{
  "dataloader": {
    "type": "async"
  },
  "dataset": {
    "extrinsics_file_path": "ARG_data/inputs/slam/images.txt",
    "folder_path": "ARG_data/inputs/images/",
    "intrinsics_file_path": "ARG_data/inputs/slam/cameras.txt",
    "extension": "png",
    "type": "png_folder"
  },
  "initializer": {
    "default_distance": 0.001,
    "enable_radius_outlier_removal": false,
    "init_opacity": 0.5,
    "init_scaling": 0.5,
    "min_distance": 1.0e-07,
    "nb_points": 16,
    "num_neighbors": 3,
    "radius": 0.05,
    "sh_degree": 3,
    "type": "knn"
  },
  "input_pc_file": "ARG_data/inputs/slam/points3D.txt",
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
    "decay_rate": 0.99769,
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
    "max_num_gaussians": 1000000,
    "max_screen_size": 20,
    "pruning_opacity_threshold": 0.005,
    "pruning_scale_threshold": 0.1,
    "refine_every": 100,
    "reset_every": 3000,
    "seed": 42,
    "start_refine": 500,
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
    "max_steps": 30000,
    "near_plane": 0.01,
    "sh_degree_interval": 1000,
    "test_steps": [3000, 7000, 30000],
    "out_dir": "outputs",
    "export_rasterized": false
  }
}
"""