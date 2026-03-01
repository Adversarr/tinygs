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

# Acknowledgements

This project is heavily inspired by
1. [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
2. [gaussian-splatting-cuda](https://github.com/MrNeRF/gaussian-splatting-cuda)

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
