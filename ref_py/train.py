from typing import Any, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import time
import os
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import imageio
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim

from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from data import CompetitionParser, CompetitionDataset
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.optimizers import SelectiveAdam

@dataclass
class CompetitionConfig:
    """Configuration for competition training."""

    # Data and paths
    data_dir: str = "/data/accgs/1751090600427/"
    result_dir: str = "results/competition"

    # Training parameters
    max_steps: int = 10_000
    batch_size: int = 1
    eval_steps: list = field(default_factory=lambda: [2_000, 5_000, 10_000])
    save_steps: list = field(default_factory=lambda: [2_000, 5_000, 10_000])

    # Model parameters
    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    init_opacity: float = 0.1
    init_scale: float = 1.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000

    # Loss parameters
    ssim_lambda: float = 0.2

    # Learning rates
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # Optimization
    sparse_grad: bool = False
    visible_adam: bool = False

    # Logging
    tb_every: int = 100
    tb_save_image: bool = False

    # Device
    device: str = "cuda"

    # Strategy
    strategy: DefaultStrategy = field(default_factory=DefaultStrategy)


def create_splats_with_optimizers(
    parser: CompetitionParser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    points = torch.from_numpy(parser.sfm_points).float()
    rgbs = torch.from_numpy(parser.sfm_colors).float()

    # # Add skybox points on a unit sphere
    # # The skybox must include all the camera and points
    # points_mean = torch.mean(points, dim=0)
    # points_radius = torch.max(torch.norm(points - points_mean, dim=1)).item()
    # camera_locations = torch.from_numpy(np.array([c2w[:3, 3] for c2w in parser.c2w_mats])).float()
    # camera_radius = torch.max(torch.norm(camera_locations - points_mean, dim=1)).item()
    # max_radius = max(points_radius, camera_radius)
    # skybox_points = len(points)
    # theta = torch.rand(skybox_points) * 2 * np.pi  # azimuthal angle
    # phi = torch.arccos(2 * torch.rand(skybox_points) - 1)  # polar angle
    # skybox_xyz = torch.stack([
    #     torch.sin(phi) * torch.cos(theta),
    #     torch.sin(phi) * torch.sin(theta), 
    #     torch.cos(phi)
    # ], dim=1) * (max_radius * 3)  # Scale the unit sphere

    # skybox_xyz = skybox_xyz + points_mean
    # skybox_colors = torch.clamp(torch.randn((skybox_points, 3)) * 0.5 + 0.5, 0, 1)  # Gray color for skybox
    # # Combine with SfM points
    # points = torch.cat([points, skybox_xyz], dim=0)
    # rgbs = torch.cat([rgbs, skybox_colors], dim=0)

    # # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]


    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    BS = batch_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizer = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizer  # type: ignore


class CompetitionTrainer:
    """Trainer for competition dataset."""

    def __init__(self, cfg: CompetitionConfig):
        self.cfg = cfg
        self.device = cfg.device

        # Setup directories
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data
        self.parser = CompetitionParser(cfg.data_dir, T_factor=1, normalize=True)
        self.dataset = CompetitionDataset(self.parser)

        # Split dataset into train/val (80/20 split)
        total_size = len(self.dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        self.trainset, self.valset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        camera_locations = np.array([c2w[:3, 3] for c2w in self.parser.c2w_mats])
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        scene_scale = np.max(dists)
        print(f"Scene center: {scene_center}, scale: {scene_scale}")
        
        print(
            f"Dataset loaded: {total_size} images ({train_size} train, {val_size} val)"
        )

        # Initialize model
        self.splats, self.optimizers = create_splats_with_optimizers(
            parser=self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opacity,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            device=self.device,
        )

        print(f"Model initialized. Number of GS: {len(self.splats['means'])}")

        # Initialize strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=scene_scale * 1.2)

        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

    def rasterize_splats(
        self,
        w2c: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        **raster_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Rasterize splats to get rendered images."""
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=w2c,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=self.cfg.strategy.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            **raster_kwargs,
        )

        return render_colors, render_alphas, info

    def train_step(self, data: Dict, **raster_kwargs) -> Dict[str, Any]:
        """Single training step."""
        # Extract data
        w2c = data['w2c'].to(self.device)
        Ks = data["K"].to(self.device)  # [1, 3, 3]
        pixels = data["image"].to(self.device)  # [1, H, W, 3]
        height, width = pixels.shape[1], pixels.shape[2]

        # Forward pass
        renders, alphas, info = self.rasterize_splats(
            w2c=w2c,
            Ks=Ks,
            width=width,
            height=height, **raster_kwargs
        )

        colors = renders
        # Compute losses
        l1loss = F.l1_loss(colors, pixels)
        # l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - self.cfg.ssim_lambda) + ssimloss * self.cfg.ssim_lambda

        return {
            "loss": loss,
            "l1loss": l1loss,
            "ssimloss": ssimloss,
            "info": info,
            "colors": colors,
            "pixels": pixels,
        }

    def train(self):
        """Main training loop."""
        cfg = self.cfg

        # Setup data loader
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Setup learning rate schedulers
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
            ),
        ]

        # Training loop
        global_tic = time.time()
        pbar = tqdm.tqdm(range(cfg.max_steps))

        for step in pbar:
            # Get next batch
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            # Spherical harmonics schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Forward pass and loss computation
            train_results = self.train_step(
                data,
                sh_degree=sh_degree_to_use,
            )
            loss = train_results["loss"]
            l1loss = train_results["l1loss"]
            ssimloss = train_results["ssimloss"]
            info = train_results["info"]

            # Pre-backward strategy step
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            # Backward pass
            loss.backward()

            # Update progress bar
            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| #g={len(self.splats['means'])}"
            pbar.set_description(desc)

            # Logging
            if step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)

                # Log gradient norms for each parameter (normalized per Gaussian)
                for param_name, param in self.splats.items():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                        num_gaussians = param.shape[0]  # First dimension is number of Gaussians
                        avg_grad_norm = grad_norm / math.sqrt(num_gaussians)
                        self.writer.add_scalar(f"train/grad_norm_{param_name}", avg_grad_norm, step)

                if cfg.tb_save_image:
                    colors = train_results["colors"][0]
                    pixels = train_results["pixels"][0]
                    canvas = torch.cat([pixels, colors], dim=1).detach().cpu().numpy()
                    self.writer.add_image("train/render", canvas, step, dataformats='HWC')
                # Calculate metrics for training visualization
                colors_p = train_results["colors"].permute(0, 3, 1, 2)  # [B, 3, H, W] 
                pixels_p = train_results["pixels"].permute(0, 3, 1, 2)  # [B, 3, H, W]
                train_psnr = self.psnr(colors_p, pixels_p)
                train_ssim = self.ssim(colors_p, pixels_p)
                
                self.writer.add_scalar("train/psnr", train_psnr, step)
                self.writer.add_scalar("train/ssim", train_ssim, step)

                self.writer.flush()

            # Optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            # Post-backward strategy step
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=False,
            )

            # Evaluation
            if step in [i - 1 for i in cfg.eval_steps] or step == cfg.max_steps - 1:
                self.eval(step)

            # Save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == cfg.max_steps - 1:
                self.save_checkpoint(step)

        print(f"Training completed in {time.time() - global_tic:.2f} seconds")

    @torch.no_grad()
    def eval(self, step: int, **raster_kwargs):
        """Evaluation on validation set."""
        print("Running evaluation...")
        
        if 'sh_degree' not in raster_kwargs:
            raster_kwargs['sh_degree'] = self.cfg.sh_degree
        
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        metrics = defaultdict(list)
        ellipse_time = 0

        for i, data in enumerate(valloader):
            w2c = data['w2c'].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device)
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            colors, _, _ = self.rasterize_splats(
                w2c=w2c,
                Ks=Ks,
                width=width,
                height=height,
                **raster_kwargs
            )

            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)

            # Save rendered images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/val_step{step}_{i:04d}.png",
                canvas,
            )

            # Compute metrics
            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))

        # Aggregate metrics
        ellipse_time /= len(valloader)
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(
            {
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
        )

        print(
            f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f} "
            f"Time: {stats['ellipse_time']:.3f}s/image "
            f"Number of GS: {stats['num_GS']}"
        )

        # Save stats
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)

        # Log to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        data = {
            "step": step,
            "splats": self.splats.state_dict(),
            "config": self.cfg,
        }
        torch.save(data, f"{self.ckpt_dir}/ckpt_{step}.pt")
        print(f"Checkpoint saved at step {step}")


if __name__ == "__main__":
    # Configuration
    cfg = CompetitionConfig(
        data_dir="/data/accgs/1748422612463/",
        result_dir="./results",
        max_steps=30000,
        eval_steps=[3000, 7000, 15000, 30000],
        save_steps=[3000, 7000, 15000, 30000],
        init_type="sfm",
        init_num_pts=100_000,
        init_extent=3.0,
        init_opacity=0.1,
        init_scale=1.0,
        sh_degree=3,
        sh_degree_interval=1000,
        means_lr=1.6e-4,
        scales_lr=5e-3,
        opacities_lr=5e-2,
        quats_lr=1e-3,
        sh0_lr=2.5e-3,
        shN_lr=1.25e-4,
        ssim_lambda=0.2,
        sparse_grad=False,
        visible_adam=False,
        batch_size=1,
        tb_every=100,
        tb_save_image=True,
        device="cuda:0",
        strategy=DefaultStrategy(verbose=True),
    )

    # Initialize trainer and start training
    trainer = CompetitionTrainer(cfg)
    trainer.train()
