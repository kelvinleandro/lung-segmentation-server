import cv2
import numpy as np

def aplicar_filtros(imagem: np.ndarray, aplicar_desfoque_media=False,
                    aplicar_desfoque_gaussiano=False,
                    aplicar_desfoque_mediana=False,
                    tamanho_kernel=5, sigma=0) -> np.ndarray:
    """
    Aplica filtros de suavização na imagem.

    args:
        imagem: np.ndarray - Imagem de entrada (em escala de cinza).
        aplicar_desfoque_gaussiano: bool - Se True, aplica o desfoque gaussiano.
        aplicar_desfoque_media: bool - Se True, aplica o desfoque de média.
        aplicar_desfoque_mediana: bool - Se True, aplica o desfoque mediano.
        tamanho_kernel: int - Tamanho ímpar do kernel para o desfoque.
        sigma: int - Valor do sigma para o desfoque gaussiano. Se for 0, o OpenCV
                     calcula automaticamente.

    return:
        np.ndarray - Imagem resultante com os filtros aplicados.
    """
    imagem_processada = imagem.copy()

    if aplicar_desfoque_gaussiano:
        imagem_processada = cv2.GaussianBlur(imagem_processada, (tamanho_kernel,
                                                                 tamanho_kernel), sigma)

    if aplicar_desfoque_media:
        imagem_processada = cv2.blur(imagem_processada, (tamanho_kernel,
                                                         tamanho_kernel))

    if aplicar_desfoque_mediana:
        imagem_processada = cv2.medianBlur(imagem_processada, tamanho_kernel)

    return imagem_processada