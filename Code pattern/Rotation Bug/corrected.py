import cv2
import numpy as np


def rotated_image(image: np.ndarray, angle: int = 45) -> np.ndarray:
    """Rotate image by angle degrees."""
    width, height, channels = image.shape
    transform = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
    result = cv2.warpAffine(image, transform, (height, width))
    return result
