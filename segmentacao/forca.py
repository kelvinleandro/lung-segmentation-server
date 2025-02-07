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
    xm = (
        pontos[(indice - 1) % len(pontos), 0] + pontos[(indice + 1) % len(pontos), 0]
    ) / 2
    ym = (
        pontos[(indice - 1) % len(pontos), 1] + pontos[(indice + 1) % len(pontos), 1]
    ) / 2
    v1 = pontos[(indice + 1) % len(pontos)] - pontos[indice]
    v2 = pontos[(indice - 1) % len(pontos)] - pontos[indice]
    vet = v1[0] * v2[1] - v1[1] * v2[0]
    sig = 1
    if vet == 0:
        return 0
    if vet < 0:
        sig = -1
    return (
        abs(pontos[indice, 0] + sig * xm) ** 2 + abs(pontos[indice, 1] + sig * ym) ** 2
    ) ** 0.5
