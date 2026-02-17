from pathlib import Path

import numpy as np
from PIL import Image


def process_images(
    input_folder: str, target_folder: str, output_size: tuple[int, int] = (512, 512)
) -> tuple[np.ndarray, np.ndarray]:
    """Process images into numpy arrays for training."""
    input_path = Path(input_folder)
    target_path = Path(target_folder)

    img_height, img_width = output_size

    input_image_array = np.empty([625, img_height, img_width, 3], dtype=np.int16)
    target_image_array = np.empty([625, img_height, img_width, 3], dtype=np.int16)

    for i in range(625):
        image_path1 = input_path / f"image_{i}.jpg"
        image1 = Image.open(image_path1)
        image_array1 = np.array(image1)
        input_image_array[i] = crop_and_limit(image_array1, img_width, img_height)

        image_path2 = target_path / f"image_{i}.jpg"
        image2 = Image.open(image_path2)
        image_array2 = np.array(image2)
        target_image_array[i] = crop_and_limit(image_array2, img_width, img_height)

    print("Images processed successfully")
    return input_image_array, target_image_array


def crop_and_limit(image_array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Crop image to specified size and limit channels."""
    cropped = image_array[:height, :width, :]
    return cropped[:, :, :3]
