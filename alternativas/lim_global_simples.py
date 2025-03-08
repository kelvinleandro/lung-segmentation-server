import numpy as np
import cv2
from segmentacao.remove_fundo import remove_fundo


def lim_global_simples(img: np.ndarray) -> tuple:
    """
    Traça o contorno do pulmão da imagem utilizando Limiarização global simples.

    args:
        img: np.ndarray - Imagem em tons de cinza
    return:
        tuple:
            - np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            - dict: Dicionário onde cada chave é uma string (e.g., "contorno_0")
                    e o valor é o contorno válido.
    """

    # imagem é filtrada com um filtro gaussiano
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # novas imagens são criadas
    img_bin = np.zeros_like(img)
    img_bin = img_bin.astype(np.uint8)

    _, img_bin = cv2.threshold(img, -500, 255, cv2.THRESH_BINARY)  # aplicada limiar
    img_bin_invertida = (255 - img_bin).astype(np.uint8)  # inverte a imagem binária

    return remove_fundo(img_bin_invertida)
