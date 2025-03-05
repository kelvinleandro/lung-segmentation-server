import cv2
import numpy as np

def aplicar_otsu(imagem_cinza: np.ndarray) -> np.ndarray:


    # Aplica um filtro Gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5,5), 0)

    # Aplica threshold de Otsu para segmentação
    _, mascara_pulmao = cv2.threshold(imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return mascara_pulmao
