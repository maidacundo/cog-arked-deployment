import cv2 
import numpy as np
from PIL import Image

def apply_canny(image, low_threshold=100, high_threshold=200):
    image_np = np.array(image)
    control_image = cv2.Canny(image_np, low_threshold, high_threshold)
    control_image = control_image[:, :, None]
    control_image = np.concatenate([control_image, control_image, control_image], axis=2)
    control_image = Image.fromarray(control_image)
    return control_image