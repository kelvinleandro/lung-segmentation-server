import numpy as np
import cv2
import numba

from segmentacao.forca import forca_adaptativa, forca_continuidade


def energia_externa(
    imagem: np.ndarray,
    probabilidade: np.ndarray,
    probablidade3: float = 0.2,
    probablidade4: float = 0.15,
) -> np.ndarray:
    """
    Recebe a imagem e as probabilidades de ocorrência de classe da imagem e retorna
    a matriz com energia externa crisp de cada ponto imagem. Pode receber os limiares
    das probabilidade P(3) e P(4) para experimentação posterior.

    Args:
        imagem (np.ndarray): Imagem para calcular a energia.
        probabilidade (np.ndarray): Probabilidade de ocorrência de classe de todos os
        pontos.
        probablidade3 (float): Limiar da probabilidade 3 para definir o valor da energia
        crispy como 0.
        probablidade4 (float): Limiar da probabilidade 4 para definir o valor da energia
        crispy como 0.
    Return:
        energia (np.ndarray): Matriz das energias crispy de todos os pontos da imagem.
    """
    # Cálculo do Sobel
    sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
    energia = np.sqrt(sobel_x**2 + sobel_y**2)

    # Mascara de probabilidade
    mask = (probabilidade[2] >= probablidade3) | (probabilidade[3] > probablidade4)
    energia[~mask] = 0

    return energia


def energia_interna_adaptativa(
    curva: np.ndarray, indice: int, w_adapt: float = 0.1, w_cont: float = 0.6
) -> np.floating:
    """
    Calcula a energia interna adaptativa, utilizando a força adaptativa e força
    de continuidade ponderadas

    Args:
        curva (np.ndarray): Curva de n pontos
        indice (int): Indice do ponto da curva
        w_adapt (float): Peso da energia interna adaptativa
        w_cont (float): Peso da energia de continuidade
    Return:
        energia_interna_adaptativa (float): Energia interna adaptativa do ponto da curva
    """
    adaptativa = w_adapt * forca_adaptativa(curva, indice)
    continuidade = w_cont * forca_continuidade(curva, indice)
    return adaptativa + continuidade


