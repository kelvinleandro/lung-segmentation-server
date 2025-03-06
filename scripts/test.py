from segmentacao.carregar import carregar_imagem
from segmentacao.contorno_ativo import MCACrisp
from PIL import Image, ImageDraw
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.DEBUG)

img = "./data/pulmao2/100.dcm"

mca = MCACrisp(
    img,
    180,
    360,
    0,
    256,
    quantidade_pixels=60,
    raio=60,
    w_cont=0.6,
    w_adapt=0.3,
    area_de_busca=9,
    d_max=5,
)


def create_gif(curves, image_path, output_gif_path, duration=100):
    """
    Create a GIF from a list of curves overlaid on the original image.

    Args:
        curves (list): List of curves (each curve is a numpy array of points).
        image_path (str): Path to the original image.
        output_gif_path (str): Path to save the output GIF.
        duration (int): Duration (in milliseconds) for each frame in the GIF.
    """
    # Load the original image
    img = carregar_imagem(image_path)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    width, height = img.size

    frames = []
    for curve in curves:
        frame = img.copy()
        draw = ImageDraw.Draw(frame)

        # Draw the curve on the frame
        if len(curve) > 0:
            points = [(point[0], point[1]) for point in curve]
            draw.line(points, fill="red", width=2)

        frames.append(frame)

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


curves = []
for curva in mca.process():
    curves.append(curva)

# Create a GIF from the collected curves
output_gif_path = "output.gif"
create_gif(curves, img, output_gif_path, duration=100)
