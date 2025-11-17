from typing import List, Union

from PIL import Image
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

import cv2

class DepthEstimator:
    """
    A reusable depth estimation module that loads the model only once
    and supports both single and batched numpy inputs.
    """

    _instance = None
    _model: DepthAnything3 = None
    _image_processor = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DepthEstimator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None or self._image_processor is None:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the depth estimation model and processor."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model from Hugging Face Hub
        model = DepthAnything3.from_pretrained("depth-anything/da3-large")
        self._model = model.to(device=self._device)
        self._model.to(self._device)
        self._model.eval()  # Set to evaluation mode

    def _numpy_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def predict_depth(
        self,
        images: Union[np.ndarray, List[np.ndarray], Image.Image, List[Image.Image]],
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predict depth for input images.

        Args:
            images: Input image(s) as numpy array(s), PIL Image(s), or list of either
                   - For numpy arrays: shape (H, W, C) or (H, W) for grayscale
                   - For batch: list of numpy arrays or PIL Images

        Returns:
            Depth map(s) as numpy array(s) with values normalized to [0, 1]
        """
        # Handle single image vs batch
        is_batch = isinstance(images, list)
        if not is_batch:
            images = [images]

        # Convert all inputs to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.ndim == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel
                pil_img = self._numpy_to_pil(img)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            pil_images.append(pil_img)

        prediction = self._model.inference(
          pil_images,
          process_res_method="lower_bound_resize",
        )
        print(prediction.is_metric, prediction.scale_factor)
        return 1 / (prediction.depth[0] + 1e-12)

    def predict_depth_as_image(
        self,
        images: Union[np.ndarray, List[np.ndarray], Image.Image, List[Image.Image]],
        normalize_to_255: bool = True,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Predict depth and return as PIL Image(s).

        Args:
            images: Input image(s) as numpy array(s), PIL Image(s), or list of either
            normalize_to_255: If True, normalize depth values to [0, 255], otherwise [0, 1]

        Returns:
            Depth map(s) as PIL Image(s)
        """
        depth_maps = self.predict_depth(images)

        is_batch = isinstance(depth_maps, list)
        if not is_batch:
            depth_maps = [depth_maps]

        depth_images = []
        for depth in depth_maps:
            if normalize_to_255:
                depth = depth * 255
            depth_img = Image.fromarray(depth.astype("uint8"))
            depth_images.append(depth_img)

        if not is_batch:
            return depth_images[0]
        return depth_images


# Example usage
if __name__ == "__main__":
    # Initialize the depth estimator (model is loaded only once)
    depth_estimator = DepthEstimator()

    # Example with a single image file
    image_path = "/data/accgs/1747834320424/inputs/images_480x640_1/0010.png"
    image = Image.open(image_path)

    # Predict depth as numpy array
    depth_array = depth_estimator.predict_depth(image)
    print(
        f"Depth array shape: {depth_array.shape}, range: [{depth_array.min():.4f}, {depth_array.max():.4f}]"
    )

    # Predict depth as PIL Image
    depth_image = depth_estimator.predict_depth_as_image(image)
    depth_image.save("depth_output.png")

    # Example with numpy array input
    image_array = np.array(image)
    depth_from_array = depth_estimator.predict_depth(image_array)
    print(f"Depth from array shape: {depth_from_array.shape}")

    # Example with batched input
    image_batch = [image, image_array]  # Mixed input types
    depth_batch = depth_estimator.predict_depth(image_batch)
    print(f"Batch depth result count: {len(depth_batch)}")

    depth_images_batch = depth_estimator.predict_depth_as_image(image_batch)
    for i, depth_img in enumerate(depth_images_batch):
        depth_img.save(f"depth_output_batch_{i}.png")
