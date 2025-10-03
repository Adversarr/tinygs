import csv
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("loss_csv", type=str, help="Path to the loss.csv file")
parser.add_argument('--logscale', action='store_true', help='Use log scale for y-axis')
# step,timestamp,l1,fused_ssim,total_loss
args = parser.parse_args()

steps = []
timestamps = []
losses = {}

with open(args.loss_csv, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        step, timestamp, l1, fused_ssim, total_loss = map(float, row)
        steps.append(step)
        timestamps.append(timestamp)
        losses.setdefault("l1", []).append(l1)
        losses.setdefault("fused_ssim", []).append(fused_ssim)
        losses.setdefault("total_loss", []).append(total_loss)

plt.figure(figsize=(10, 6))
plt.plot(steps, losses["l1"], label="L1")
plt.plot(steps, losses["fused_ssim"], label="Fused SSIM")
plt.plot(steps, losses["total_loss"], label="Total Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
if args.logscale:
    plt.yscale("log")
plt.legend()
plt.title("Training Loss")
plt.savefig("loss.png")
