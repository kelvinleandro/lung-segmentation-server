import numpy as np


def forca_continuidade(pontos: np.ndarray, indice: int) -> np.float64:
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


def forca_adaptativa(pontos: np.ndarray, indice: int) -> np.float64:
    """
    Recebe os pontos em ordem anti-horária da curva e calcula a força adaptativa
    no ponto de índice 'indice'.

    Args:
        pontos (np.ndarray): Pontos da curva.
        indice (int): Índice do ponto.
    Returns:
        float: Força adaptativa no ponto de índice 'indice'.
    """
    pm = (pontos[(indice - 1) % len(pontos)] + pontos[(indice + 1) % len(pontos)]) / 2
    v1 = pontos[(indice + 1) % len(pontos)] - pontos[indice]
    v2 = pontos[(indice - 1) % len(pontos)] - pontos[indice]
    vet = np.sign(np.linalg.det([v1, v2]))
    if vet == 0:
        return np.float64(0.0)
    return np.linalg.norm(pm + vet * pontos[indice])
