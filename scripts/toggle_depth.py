import da3_depth as depth
import numpy as np

def toggle():
    input_images = np.random.rand(256, 256, 3).astype(np.float32)
    depth_estimator = depth.DepthEstimator()
    depth_estimator.predict_depth(input_images)
    print("✅ Depth anything setup!")

if __name__ == '__main__':
    toggle()