import argparse
import os
import sys
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Batch launcher for SIGA2 training')
    parser.add_argument('--root-dir', type=str, required=True, help='Path to your root of SIGA2 dataset.')
    parser.add_argument('--working-dir', type=str, required=True, help='Working directory to store preprocessed data and outputs.')
    parser.add_argument('--magic-number', type=int, default=42, help='Magic number to prepend to the UUID of each image.')
    parser.add_argument('--log-level', type=str, default='warn', choices=['debug', 'info', 'warn', 'error'], help='Verbose level of logger.')
    parser.add_argument('--normalize', action='store_true', help='Normalize the points3D to have std=1.')
    parser.add_argument('--use-hf-mirror', action='store_true', help='Use the HF mirror for DepthAnything.')
    parser.add_argument('--align', action='store_true', help='Align the points3D with DepthAnything.')
    return parser.parse_args()

def _find_config_train(repo_root: Path) -> Path:
    exe = repo_root / 'config_train'
    if exe.exists():
        return exe
    # Fallbacks if root exe not present
    for cand in [repo_root / 'build' / 'Release' / 'examples' / 'config_train',
                 repo_root / 'build' / 'Debug' / 'examples' / 'config_train']:
        if cand.exists():
            return cand
    raise FileNotFoundError('config_train executable not found. Please run build.sh first.')

def main():
    args = parse_args()
    if args.use_hf_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    repo_root = Path(__file__).parent.parent
    root_dir = Path(args.root_dir)
    work_root = Path(args.working_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    config_train_bin = None
    try:
        config_train_bin = _find_config_train(repo_root)
    except Exception as e:
        print(f'❌ {e}')

    scenes = []
    if root_dir.exists() and root_dir.is_dir():
        scenes = [p.name for p in root_dir.iterdir() if p.is_dir() and 'readme' not in p.name.lower()]
        scenes.sort()
    else:
        print(f'❌ Root dir {root_dir} does not exist or is not a directory.')
        return

    failed = {}
    for i, sid in enumerate(scenes):
        print("=" * 80)
        print(f'🚀 Processing scene {sid}')
        print("=" * 80)
        # Preprocess
        try:
            subargs = [sys.executable, str(repo_root / 'scripts' / 'preprocess_siga2.py'),
                       '--input', str(root_dir), '--id', sid, '--output', str(work_root),
                       '--magic-number', str(args.magic_number)]
            if args.normalize or args.align: # Aligner requires normalized points3D
                subargs.append('--normalize')
            subprocess.run(subargs, check=True)
        except subprocess.CalledProcessError as e:
            print(f'❌ Preprocess failed for {sid}: {e}')
            failed[sid] = 'preprocess'
            continue

        scene_dir = work_root / sid
        cfg_path = scene_dir / 'config.json'

        aligner_success = False
        time_budget = 59
        if args.align:
            try:
                subprocess.run([sys.executable, str(repo_root / 'scripts' / 'aligner_siga2.py'),
                                '--working_dir', str(scene_dir)], check=True)
                time_budget = int(Path(scene_dir / 'aligner_time.txt').read_text().strip())
                aligner_success = True
            except subprocess.CalledProcessError as e:
                print(f'❌ Aligner failed for {sid}: {e}')
                failed[sid] = 'aligner'
                continue

        # Generate config via gen_config_siga2.py
        try:
            subargs = [sys.executable, str(repo_root / 'scripts' / 'gen_config_siga2.py'),
                       '--working_dir', str(scene_dir), '--time-budget', str(time_budget), '--out', str(cfg_path)]
            if aligner_success:
                if Path(scene_dir / 'aligned_points.ply').exists():
                    subargs.append('--use-aligned')
                else:
                    print(f'⚠️ Aligner did not produce aligned_points.ply for {sid}, skipping --use-aligned')
            subprocess.run(subargs, check=True)
        except subprocess.CalledProcessError as e:
            print(f'❌ Config generation failed for {sid}: {e}')
            failed[sid] = 'gen_config'
            continue

        print("=" * 80)
        print(f'🚀 Training scene {sid}')
        print("=" * 80)
        # Train
        if config_train_bin is None:
            print(f'❌ config_train not available, skipping training for {sid}')
            failed[sid] = 'config_train_missing'
            continue
        try:
            cmd = [str(config_train_bin), '-c', str(cfg_path), '-l', args.log_level]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f'❌ Training failed for {sid}: {e}')
            failed[sid] = 'train'
            continue
        print(f"✅ Training completed for {sid} ({i + 1}/{len(scenes)})")
    # Final report regardless of individual failures
    try:
        subprocess.run([sys.executable, str(repo_root / 'scripts' / 'report_final_siga2.py'),
                        '--working_dir', str(work_root)], check=True)
    except subprocess.CalledProcessError as e:
        print(f'❌ Final reporting failed: {e}')

    if failed:
        print('⚠️ Some scenes failed:')
        for k, v in failed.items():
            print(f'  - {k}: {v}')
    
    print("✅ All scenes processed!")

if __name__ == '__main__':
    main()