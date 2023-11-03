import cv2 
import numpy as np
from PIL import Image

def crop(image):
    height = (image.height // 8) * 8
    width = (image.width // 8) * 8
    left = int((image.width - width) / 2)
    right = left + width
    top = int((image.height - height) / 2)
    bottom = top + height
    image = image.crop((left, top, right, bottom))
    return image

def scale_down_image(self, image_path, max_size):
    #Open the Image
    image = Image.open(image_path)
    #Get the Original width and height
    width, height = image.size
    # Calculate the scaling factor to fit the image within the max_size
    scaling_factor = min(max_size/width, max_size/height)
    # Calaculate the new width and height
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    #resize the image
    resized_image = image.resize((new_width, new_height))
    cropped_image = self.crop_center(resized_image)
    return cropped_image

def crop_center(self, pil_img):
    img_width, img_height = pil_img.size
    crop_width = self.base(img_width)
    crop_height = self.base(img_height)
    return pil_img.crop(
            (
                (img_width - crop_width) // 2,
                (img_height - crop_height) // 2,
                (img_width + crop_width) // 2,
                (img_height + crop_height) // 2)
            )

def base(self, x):
    return int(8 * math.floor(int(x)/8))

def apply_canny(image, low_threshold=100, high_threshold=200):
    image_np = np.array(image)
    control_image = cv2.Canny(image_np, low_threshold, high_threshold)
    control_image = control_image[:, :, None]
    control_image = np.concatenate([control_image, control_image, control_image], axis=2)
    control_image = Image.fromarray(control_image)
    return control_image