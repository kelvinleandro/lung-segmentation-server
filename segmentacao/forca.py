import numpy as np


def forca_continuidade(pontos: np.array, indice: int) -> np.float64:
    """
    Calcula a força de continuidade da curva em um ponto.

    Args:
        pontos: np.array - Pontos da curva.
        indice: int - Índice do ponto.
    Return:
        float - Força de continuidade.
    """

    # Calcular a distância média entre os pontos
    dm = np.linalg.norm(pontos - np.roll(pontos, 1, axis=0), axis=1).mean()

    # Calcular a derivada discreta do ponto
    dc = np.linalg.norm(pontos[indice] - pontos[indice - 1])

    return np.abs(dm - dc)

