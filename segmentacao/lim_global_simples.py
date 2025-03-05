import numpy as np
import cv2

def lim_global_simples(img: np.ndarray) -> np.ndarray:
    """
    Limiarização global simples.

    args:
        img: np.ndarray - Imagem em tons de cinza
    return:
        np.ndarray - Imagem limiasrizada
    """
    img = cv2.GaussianBlur(img, (5,5), 0)
    img_bin = np.zeros_like(img)
    _, img_bin = cv2.threshold(img, -500, 255, cv2.THRESH_BINARY)
    img_bin = np.uint8(img_bin * 255)
    img = np.zeros_like(img)
    img = img.astype(np.uint8)
    contornos, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contornos_filtrados = [cnt for cnt in contornos if cv2.contourArea(cnt) > 1200]
    img_limiarizada = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_limiarizada, contornos_filtrados, -1, (0, 0, 255), 1)
    return img_limiarizada