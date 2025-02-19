import cv2
import numpy as np
import argparse

from segmentacao.carregar import carregar_imagem


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
            cv2.imshow(f"image_{i}", image)
            key = cv2.waitKey(0)
            if key == ord("q"):
                cv2.destroyAllWindows()
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_de_entrada", help="Caminho para arquivo DICOM com pulmão saudável"
    )
    args = parser.parse_args()

    image = carregar_imagem(args.image_de_entrada)
    print(image.max())
    visualize([apply_window(image, -300, 700)])
    cv2.destroyAllWindows()
