import argparse
import sys
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Batch launcher for SIGA2 training')
    parser.add_argument('--root-dir', type=str, required=True, help='Path to your root of SIGA2 dataset.')
    parser.add_argument('--working-dir', type=str, required=True, help='Working directory to store preprocessed data and outputs.')
    parser.add_argument('--magic-number', type=int, default=42, help='Magic number to prepend to the UUID of each image.')
    parser.add_argument('--log-level', type=str, default='warn', choices=['debug', 'info', 'warn', 'error'], help='Verbose level of logger.')
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
    repo_root = Path(__file__).parent.parent
    root_dir = Path(args.root_dir)
    work_root = Path(args.working_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    config_train_bin = None
    try:
        config_train_bin = _find_config_train(repo_root)
    except Exception as e:
        print(f'[ERROR] {e}')

    scenes = []
    if root_dir.exists() and root_dir.is_dir():
        scenes = [p.name for p in root_dir.iterdir() if p.is_dir() and 'readme' not in p.name.lower()]
        scenes.sort()
    else:
        print(f'[ERROR] Root dir {root_dir} does not exist or is not a directory.')
        return

    failed = {}
    for sid in scenes:
        print("=" * 80)
        print(f'[INFO] Processing scene {sid}')
        # Preprocess
        try:
            subprocess.run([sys.executable, str(repo_root / 'scripts' / 'preprocess_siga2.py'),
                            '--input', str(root_dir), '--id', sid, '--output', str(work_root),
                            '--magic-number', str(args.magic_number)], check=True)
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] Preprocess failed for {sid}: {e}')
            failed[sid] = 'preprocess'
            continue

        scene_dir = work_root / sid
        cfg_path = scene_dir / 'config.json'

        # Generate config via gen_config_siga2.py
        try:
            subprocess.run([sys.executable, str(repo_root / 'scripts' / 'gen_config_siga2.py'),
                            '--working_dir', str(scene_dir), '--out', str(cfg_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] Config generation failed for {sid}: {e}')
            failed[sid] = 'gen_config'
            continue

        print("=" * 80)
        print(f'[INFO] Training scene {sid}')
        print("=" * 80)
        # Train
        if config_train_bin is None:
            print(f'[ERROR] config_train not available, skipping training for {sid}')
            failed[sid] = 'config_train_missing'
            continue
        try:
            cmd = [str(config_train_bin), '-c', str(cfg_path), '-l', args.log_level]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] Training failed for {sid}: {e}')
            failed[sid] = 'train'
            continue

    # Final report regardless of individual failures
    try:
        subprocess.run([sys.executable, str(repo_root / 'scripts' / 'report_final_siga2.py'),
                        '--working_dir', str(work_root)], check=True)
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] Final reporting failed: {e}')

    if failed:
        print('[WARN] Some scenes failed:')
        for k, v in failed.items():
            print(f'  - {k}: {v}')

if __name__ == '__main__':
    main()