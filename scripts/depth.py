from typing import List, Union

from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

class DepthEstimator:
    """
    A reusable depth estimation module that loads the model only once
    and supports both single and batched numpy inputs.
    """

    _instance = None
    _model = None
    _image_processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DepthEstimator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None or self._image_processor is None:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the depth estimation model and processor."""
        self._image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self._model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
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

        # Prepare target sizes for post-processing
        target_sizes = [(img.height, img.width) for img in pil_images]

        # Process images
        inputs = self._image_processor(images=pil_images, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process depth predictions
        post_processed_output = self._image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=target_sizes,
        )

        # Extract and normalize depth maps
        depth_maps = []
        for i, output in enumerate(post_processed_output):
            predicted_depth = output["predicted_depth"]
            depth = predicted_depth.detach().cpu().numpy()
            depth_maps.append(depth)

        # Return single array if input was not a batch
        if not is_batch:
            return depth_maps[0]
        return depth_maps

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
