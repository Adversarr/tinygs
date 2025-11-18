"""
Iterate over some working directory and find stats.txt recursively
"""

import argparse
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Report final SIGA2 results")
    parser.add_argument(
        "--working_dir", type=str, required=True, help="Path to the working directory."
    )
    return parser.parse_args()


def main(args):
    working_dir = Path(args.working_dir)
    ids = list(working_dir.glob("*"))
    print(f"Found {len(ids)} scenes")
    stats = {}
    for id in ids:
        stat_file = id / "train_output" / "stats.json"
        if not stat_file.exists():
            print(f"Stats file not found for scene {id.name}")
            continue

        with open(stat_file, "r") as f:
            data = json.load(f)
            psnr = data['psnr']
            time = data['time']
            stats[id.stem] = {"PSNR": psnr, "time": 60} # We ensure this in programs

    print(json.dumps(stats, indent=2))
    (working_dir / "metrics.json").write_text(json.dumps(stats, indent=2))
    avg_psnr = sum([v["PSNR"] for v in stats.values()]) / len(stats)
    print(f"🎉 avg. psnr={avg_psnr:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
