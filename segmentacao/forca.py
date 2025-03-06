import numpy as np
from numba import jit

from segmentacao.curva import na_curva


@jit(nopython=True)
def forca_continuidade(pontos: np.ndarray, indice: int) -> float:
    """
    Calcula a força de continuidade da curva em um ponto.

    Args:
        pontos: np.array - Pontos da curva.
        indice: int - Índice do ponto.
    Return:
        float - Força de continuidade.
    """
    tam = pontos.shape[0]

    # Criar um array deslocado manualmente
    pontos_shifted = np.empty_like(pontos)
    pontos_shifted[1:] = pontos[:-1]  # Deslocamento equivalente ao np.roll
    pontos_shifted[0] = pontos[-1]  # Circularidade mantida

    # Calcular a distância média entre os pontos
    dm = np.mean(np.sqrt(np.sum((pontos - pontos_shifted) ** 2, axis=1)))

    # Calcular a derivada discreta do ponto
    dc = np.sqrt(np.sum((pontos[indice] - pontos[indice - 1]) ** 2))

    return abs(dm - dc)


@jit(nopython=True)
def forca_adaptativa(pontos: np.ndarray, indice: int) -> float:
    """
    Recebe os pontos em ordem anti-horária da curva e calcula a força adaptativa
    no ponto de índice 'indice'.

    Args:
        pontos (np.ndarray): Pontos da curva.
        indice (int): Índice do ponto.
    Returns:
        float: Força adaptativa no ponto de índice 'indice'.
    """
    tam = len(pontos)
    pm = (pontos[(indice - 1) % tam] + pontos[(indice + 1) % tam]) / 2
    sign = 0

    if na_curva(pm, pontos):
        sign = 1
    else:
        sign = -1

    # v1 = pontos[(indice + 1) % tam] - pontos[indice]
    # v2 = pontos[(indice - 1) % tam] - pontos[indice]
    # det_v1_v2 = v1[0] * v2[1] - v1[1] * v2[0]  # Determinante 2D manual
    # vet = np.sign(det_v1_v2)
    #
    # if vet == 0:
    #     return 0.0

    return np.sqrt(np.sum((pm + sign * pontos[indice]) ** 2))
