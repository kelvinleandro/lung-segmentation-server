import numpy as np


def inicializa_curva(
    ponto: np.ndarray,
    raio: int = 30,
    quantidade_pixels: int = 30,
) -> np.ndarray:
    """
    Recebe o ponto inicial do eixo x e y, calcula a curva e retorna a duas
    listas de pontos que representam a curva.

    Args:
        ponto: (x,y) que representa o centro da curva
        raio: raio da curva
        quantidade_pixels: quantidade de pontos que a curva terá
    Return:
        curva: lista de pontos que representam a curva
    """
    angulos = np.linspace(0, 2 * np.pi, quantidade_pixels)
    curva = ponto + np.c_[np.cos(angulos), np.sin(angulos)] * raio

    return curva.astype(np.int16)
