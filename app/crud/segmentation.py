from typing import List, Tuple
import numpy as np



def segment_image(pixel_array) -> List[Tuple[float, float]]:
    """
    Funcao experimental para testar a api localmente (a partir daqui sera implementada a funcao de segmentacao)

    """
    points = []

    height, width = pixel_array.shape
    for i in range(0, height, int(height / 5)):
        for j in range(0, width, int(width / 5)):
            points.append((float(i), float(j)))

    return points


def segment_image2(pixel_array, p1, p2, p3) -> List[Tuple[float, float]]:
    """
    Funcao experimental para testar a api localmente (a partir daqui sera implementada a funcao de segmentacao)

    """
    points = []

    height, width = pixel_array.shape
    for i in range(0, height, int(height / 11)):
        for j in range(0, width, int(width / 11)):
            points.append((float(i), float(j)))

    return points


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
    Raises:
        AssertionError: caso curva seja menor que 0 em algum ponto
        AssertionError: caso a quantidade_pixels seja menor que 2
    """
    assert quantidade_pixels >= 2, "Quantidade de pixels deve ser sempre maior que 2"

    angulos = np.linspace(0, 2 * np.pi, quantidade_pixels+1)
    curva = ponto + np.c_[np.cos(angulos), np.sin(angulos)] * raio

    assert np.all(curva > 0), "Os pontos da curva devem sempre ser maior que 0"

    return curva[:-1].astype(np.int16)  