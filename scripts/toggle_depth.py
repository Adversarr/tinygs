import depth
import numpy as np

input_images = np.random.rand(256, 256, 3).astype(np.float32)
depth_estimator = depth.DepthEstimator()
depth_map = depth_estimator.predict_depth(input_images)

print("✅ Depth anything setup!")
