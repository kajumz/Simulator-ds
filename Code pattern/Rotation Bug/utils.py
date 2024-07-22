import cv2
import numpy as np
from skimage.data import coffee
from skimage import io
from matplotlib import pyplot as plt
cof = coffee()
io.imshow(cof)
plt.show()
#print(coffee().shape)
print(cof.shape)
def rotated_image(image: np.ndarray, angle: int = 45) -> np.ndarray:
    """Rotate image by angle degrees."""
    width, height, channels = image.shape
    transform = cv2.getRotationMatrix2D((height / 2, width / 2), angle, 1)
    result = cv2.warpAffine(image, transform, (height, width))
    return result
rot = rotated_image(cof, 90)
print(rot.shape)
io.imshow(rot)
plt.show()