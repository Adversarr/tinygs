import csv
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
from scipy.ndimage import gaussian_filter1d

parser = ArgumentParser()
parser.add_argument("loss_csv", type=str, help="Path to the loss.csv file")
parser.add_argument('--logscale', action='store_true', help='Use log scale for y-axis')
parser.add_argument('--smoothing', type=float, default=2.0, help='Smoothing factor for Gaussian filter (higher = more smoothing)')
# step,timestamp,fused_ssim,l1,total_loss
args = parser.parse_args()

steps = []
timestamps = []
losses = {}

with open(args.loss_csv, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        step, timestamp, fused_ssim, l1, total_loss = map(float, row)
        steps.append(step)
        timestamps.append(timestamp)
        losses.setdefault("l1", []).append(l1)
        losses.setdefault("fused_ssim", []).append(fused_ssim)
        losses.setdefault("total_loss", []).append(total_loss)


print(np.mean(losses["total_loss"]), np.mean(losses["l1"]), np.mean(losses["fused_ssim"]))

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# L1 Loss subplot
axes[0].plot(steps, losses["l1"], alpha=0.5, label="L1 (Original)")
smoothed_l1 = gaussian_filter1d(losses["l1"], sigma=args.smoothing)
axes[0].plot(steps, smoothed_l1, label=f"L1 (Smoothed, σ={args.smoothing})")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].grid(True)
if args.logscale:
    axes[0].set_yscale("log")
axes[0].legend()
axes[0].set_title("L1 Training Loss")

# Fused SSIM Loss subplot
axes[1].plot(steps, losses["fused_ssim"], alpha=0.5, label="Fused SSIM (Original)")
smoothed_fused_ssim = gaussian_filter1d(losses["fused_ssim"], sigma=args.smoothing)
axes[1].plot(steps, smoothed_fused_ssim, label=f"Fused SSIM (Smoothed, σ={args.smoothing})")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss")
axes[1].grid(True)
if args.logscale:
    axes[1].set_yscale("log")
axes[1].legend()
axes[1].set_title("Fused SSIM Training Loss")

# Total Loss subplot
axes[2].plot(steps, losses["total_loss"], alpha=0.5, label="Total Loss (Original)")
smoothed_total_loss = gaussian_filter1d(losses["total_loss"], sigma=args.smoothing)
axes[2].plot(steps, smoothed_total_loss, label=f"Total Loss (Smoothed, σ={args.smoothing})")
axes[2].set_xlabel("Step")
axes[2].set_ylabel("Loss")
axes[2].grid(True)
if args.logscale:
    axes[2].set_yscale("log")
axes[2].legend()
axes[2].set_title("Total Training Loss")

plt.tight_layout()
plt.savefig("loss.png")
