import cv2
import numpy as np
import base64


def imagem_para_base64(imagem: np.ndarray) -> str:
    """Converte uma imagem em um array NumPy para uma string Base64."""
    # Codificar a imagem em formato PNG (ou outro formato de sua escolha)
    _, buffer = cv2.imencode('.png', imagem)
    imagem_base64 = base64.b64encode(buffer).decode('utf-8')
    return imagem_base64