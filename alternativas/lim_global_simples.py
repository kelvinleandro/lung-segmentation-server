import numpy as np
import cv2
from segmentacao.carregar import carregar_imagem

def lim_global_simples(img: np.ndarray) -> np.ndarray:
    """
    Limiarização global simples.

    args:
        img: np.ndarray - Imagem em tons de cinza
    return:
        np.ndarray - Imagem limiarizada
    """

    # imagem é filtrada com um filtro gaussiano
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # novas imagens são criadas
    img_bin = np.zeros_like(img)

    _, img_bin = cv2.threshold(img, -500, 255, cv2.THRESH_BINARY)  # aplicada limiar
    img = np.zeros_like(img)
    img_bin = np.uint8(img_bin * 255)
    img = img.astype(np.uint8)  # construção de nova imagem

    # contornos traçados
    contornos, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contornos filtrados
    contornos_filtrados = [cnt for cnt in contornos if cv2.contourArea(cnt) > 1200]
    contornos_filtrados = sorted(contornos_filtrados, key=cv2.contourArea, reverse=True)

    # contorno do corpo é removido
    if len(contornos_filtrados) > 2:
        contornos_filtrados = contornos_filtrados[1:3]

    # contorno é desenhado na imagem
    img_limiarizada = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_limiarizada, contornos_filtrados, -1, (0, 0, 255), 1)
    img_limiarizada = cv2.cvtColor(img_limiarizada, cv2.COLOR_BGR2GRAY)

    return img_limiarizada

