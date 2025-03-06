import cv2
import numpy as np
from crud.alternativas.remove_fundo import remove_fundo


def aplicar_otsu(imagem: np.ndarray) -> tuple:
    """
    Aplica o algoritmo de Otsu para segmentação dos pulmões em imagens.

    Parâmetros:
        imagem (np.ndarray): Pixels da imagem de entrada em escala de cinza.

    Retorna:
        tuple:
            - Imagem original com os contornos dos pulmões destacados em azul.
            - Imagem com apenas os contornos dos pulmões em azul sobre fundo preto.

    Resumo da teoria:
        O método é uma variação da técnica de limiarização em imagens tons de cinza.
        Ele calcula um limiar ótimo minimizando a variância intra-classe (dentro das regiões)
        e maximizando a variância inter-classe (entre as regiões). É particularmente eficaz para imagens bimodais,
        onde há dois picos bem definidos no histograma.
    """

    # Aplicar threshold de Otsu para segmentação
    _, mascara_pulmao = cv2.threshold(
        imagem, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return remove_fundo(mascara_pulmao)