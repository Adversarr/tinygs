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


## Requirements

We have tested tinygs on the following platforms:
1. Ubuntu 22.04 / Manjaro Linux (Latest). (No Windows & macOS support yet.)
2. CUDA 12.4+ (12.6 to 12.9 is the recommended version)
3. gcc 11.4.0

> We provide the output of our system here to help you reproduce the results.

**My PC:**
```sh
> uname -a
Linux adversarr-ROG-Desktop 6.12.48-1-MANJARO #1 SMP PREEMPT_DYNAMIC Fri, 19 Sep 2025 16:11:04 +0000 x86_64 GNU/Linux

> gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-pc-linux-gnu/15.2.1/lto-wrapper
Target: x86_64-pc-linux-gnu
Configured with: /build/gcc/src/gcc/configure --enable-languages=ada,c,c++,d,fortran,go,lto,m2,objc,obj-c++,rust,cobol --enable-bootstrap --prefix=/usr --libdir=/usr/lib --libexecdir=/usr/lib --mandir=/usr/share/man --infodir=/usr/share/info --with-bugurl=https://gitlab.archlinux.org/archlinux/packaging/packages/gcc/-/issues --with-build-config=bootstrap-lto --with-linker-hash-style=gnu --with-system-zlib --enable-__cxa_atexit --enable-cet=auto --enable-checking=release --enable-clocale=gnu --enable-default-pie --enable-default-ssp --enable-gnu-indirect-function --enable-gnu-unique-object --enable-libstdcxx-backtrace --enable-link-serialization=1 --enable-linker-build-id --enable-lto --enable-multilib --enable-plugin --enable-shared --enable-threads=posix --disable-libssp --disable-libstdcxx-pch --disable-werror
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 15.2.1 20250813 (GCC)

> nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

**Our GPU Server:**

```sh
> uname -a
Linux gpuserver 6.8.0-64-generic #67~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Jun 24 15:19:46 UTC 2 x86_64 x86_64 x86_64 GNU/Linux

> gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-linux-gnu/11/lto-wrapper
OFFLOAD_TARGET_NAMES=nvptx-none:amdgcn-amdhsa
OFFLOAD_TARGET_DEFAULT=1
Target: x86_64-linux-gnu
Configured with: ../src/configure -v --with-pkgversion='Ubuntu 11.4.0-1ubuntu1~22.04.2' --with-bugurl=file:///usr/share/doc/gcc-11/README.Bugs --enable-languages=c,ada,c++,go,brig,d,fortran,objc,obj-c++,m2 --prefix=/usr --with-gcc-major-version-only --program-suffix=-11 --program-prefix=x86_64-linux-gnu- --enable-shared --enable-linker-build-id --libexecdir=/usr/lib --without-included-gettext --enable-threads=posix --libdir=/usr/lib --enable-nls --enable-bootstrap --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --with-default-libstdcxx-abi=new --enable-gnu-unique-object --disable-vtable-verify --enable-plugin --enable-default-pie --with-system-zlib --enable-libphobos-checking=release --with-target-system-zlib=auto --enable-objc-gc=auto --enable-multiarch --disable-werror --enable-cet --with-arch-32=i686 --with-abi=m64 --with-multilib-list=m32,m64,mx32 --enable-multilib --with-tune=generic --enable-offload-targets=nvptx-none=/build/gcc-11-2Y5pKs/gcc-11-11.4.0/debian/tmp-nvptx/usr,amdgcn-amdhsa=/build/gcc-11-2Y5pKs/gcc-11-11.4.0/debian/tmp-gcn/usr --without-cuda-driver --enable-checking=release --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --with-build-config=bootstrap-lto-lean --enable-link-serialization=2
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.2)

> /usr/local/cuda/bin/nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

Please make sure you have installed the required dependencies before running the project.

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
