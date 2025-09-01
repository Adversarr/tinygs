# tinygs: Reconstruct your scene with 3DGS in seconds.

tinygs is a lightweight library for reconstructing 3D scenes with 3DGS in seconds.

Features:
1. C++/CUDA with c++20 support.
2. Fused operations for efficient training and inference.
3. Easy-to-use API for scene reconstruction.
4. fp16 support for faster training.

Known Issues:
1. single GPU support only.
2. no Python interface yet.
3. not optimized for other platforms, only Linux is supported.

# Acknowledgements

This project is heavily inspired by
1. [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
2. [gaussian-splatting-cuda](https://github.com/MrNeRF/gaussian-splatting-cuda)