# tinygs: Reconstruct your scene with 3DGS in seconds.

tinygs is a lightweight library for reconstructing 3D scenes with 3DGS in seconds.

Features:
1. C++/CUDA with c++20 support.
2. Fused operations for efficient training and inference.
3. fp16 support for faster training. (WIP)

Known Issues:
1. single GPU support only.
2. no Python interface yet.
3. batch_size=1 is a must.
4. not optimized for other platforms, only Linux is supported.

## Setup and Run the competetion (SIGA 2025)

```bash
# install dependencies for python part.
uv sync
source .venv/bin/activate
# build the project with cmake
bash build.sh
# Per scene run
bash launch.sh  PATH_TO_DATASETS SCENE_ID # run the project with the dataset and scene id.
```

example:

```
bash launch.sh /data/SIGA_Competetion/Final/ 1750383597053
```

## Script Usage

Build script (`build.sh`)

```bash
./build.sh --help
# Example with overrides:
NVCC=/usr/local/cuda/bin/nvcc BUILD_TYPE=Release JOBS=$(nproc) TARGETS="video_to_png config_train" ./build.sh
```

Environment variables:
- `NVCC`: path to `nvcc` if not in `PATH`
- `BUILD_TYPE`: `Release` (default) or `Debug`
- `JOBS`: parallel jobs (defaults to `nproc`)
- `TARGETS`: space-separated CMake targets (default: `video_to_png config_train`)

Launch script (`launch.sh`)

```bash
./launch.sh --help
./launch.sh <root> <scene_id> [output_dir]
# Example with overrides:
PYTHON=python3 VIDEO_TO_PNG=./video_to_png CONFIG_TRAIN=./config_train ./launch.sh /data my_scene out
```

Config notes

- `trainer.max_seconds`: stops training after the given seconds (0 disables).

Arguments:
- `root`: dataset root containing the scene folder
- `scene_id`: scene folder name under `root`
- `output_dir`: output directory (default: `output`)

Environment variables:
- `VIDEO_TO_PNG`: path to the `video_to_png` executable (default: `./video_to_png`)
- `CONFIG_TRAIN`: path to the `config_train` executable (default: `./config_train`)
- `PYTHON`: Python interpreter (default: `python`)

# Acknowledgements

This project is heavily inspired by
1. [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
2. [gaussian-splatting-cuda](https://github.com/MrNeRF/gaussian-splatting-cuda)

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
