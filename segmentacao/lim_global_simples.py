import numpy as np
import cv2
from carregar import carregar_imagem
import pydicom as dicom
import matplotlib.pylab as plt

def lim_global_simples(img: np.ndarray) -> np.ndarray:
    """
    Limiarização global simples.

    args:
        img: np.ndarray - Imagem em tons de cinza
    return:
        np.ndarray - Imagem limiarizada
    """
    img_limiarizada = np.zeros_like(img)
    img_limiarizada[img > 500] = 255
    return img_limiarizada

img = dicom.dcmread('data/pulmao2/1.dcm')

plt.imshow(img.pixel_array)