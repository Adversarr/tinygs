# TODOs

## General

**CMake**: many flags are not set properly in nvcc.

**Strategy**:
1. mcmc: should tune the noise, and pruning.

**Configuration**:
1. json support, cxxopts

**Dataset**:
1. Load the dataset from video.

## Trainer

1. Implement the training loop class.
2. Pose optimization

## LR scheduler

1. Global scheduler: exponentially decay the learning rate?
2. Locally, use a scheduler for each gaussian?

## Fatals in Default Strategy

I don't know why it is incorrect.

## AoS

1. Make fastgs internally use AoS to store the 2D gaussians.
2. Make the overall codebase use AoS to store the 3D gaussians.

## Mixed precision training (Low priority) 

1. fwd, bwd, ...
2. gradient scaler