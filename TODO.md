# TODOs

## Trainer

1. Implement the training loop class.

## LR scheduler

1. Global scheduler: exponentially decay the learning rate.
2. Locally, use a scheduler for each gaussian?

## AoS

1. Make fastgs internally use AoS to store the 2D gaussians.
2. Make the overall codebase use AoS to store the 3D gaussians.

## Mixed precision training (Low priority) 

1. fwd, bwd, ...
2. gradient scaler