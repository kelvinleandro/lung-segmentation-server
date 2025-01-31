import cv2
import numpy as np
import sys
import argparse

from segmentation.loading import load_image

def apply_window(image, window_center, window_width):
    """Applies windowing to enhance contrast in the DICOM image."""
    img_min = window_center - (window_width // 2)
    img_max = window_center + (window_width // 2)
    
    image = np.clip(image, img_min, img_max)
    
    image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    return image

def visualize(images: list[cv2.typing.MatLike]):
    """
    Visualizes a list of images using OpenCV.

    args:
        images: List of images to be displayed.
    """
    while True:
        for i, image in enumerate(images):
            cv2.imshow(f"image_i", image)
            key = cv2.waitKey(0)
            if key == ord("q"):
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", help="Path to DICOM lung image")
    args = parser.parse_args()

    image = load_image(args.input_image)
    print(image.max())
    visualize([apply_window(image, -300, 700)])
    cv2.destroyAllWindows()
