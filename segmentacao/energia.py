import numpy as np
from carregar import carregar_imagem
from classificacao import calcula_ocorrencias_classes
import cv2


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
        probablidade3 (int): Limiar da probabilidade 3 para definir o valor da energia
        crispy
        como 0.
        probablidade4 (int): Limiar da probabilidade 4 para definir o valor da energia
        crispy
        como 0.
    Return:
        energia (float): Matriz com as energias crispy de todos os pontos da imagem.
    """
    energia = np.zeros(imagem.shape)
    for (x, y), value in np.ndenumerate(energia):
        if not (
            probabilidade[2][x][y] < probablidade3
            and probabilidade[3][x][y] <= probablidade4
        ):
            energia[x][y] = S(imagem, x, y)

    return energia


def S(imagem: np.ndarray, x: int, y: int) -> float:
    """
    Calcula a magnitude do gradiente de Sobel no ponto (x, y) da imagem.

    Parâmetros:
        imagem (np.ndarray): A imagem em escala de cinza.
        x (int): Coordenada x do ponto.
        y (int): Coordenada y do ponto.

    Retorna:
        magnitude (float): A magnitude do gradiente de Sobel no ponto (x, y).
    """

    if x < 0 or y < 0 or x >= imagem.shape[1] or y >= imagem.shape[0]:
        raise ValueError("Coordenadas (x, y) devem estar nos limites da imagem.")
    sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x = sobel_x[y, x]
    gradient_y = sobel_y[y, x]
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return magnitude


if "__main__":
    img = carregar_imagem("data/pulmao2/30.dcm")
    print(type(img))
    func = calcula_ocorrencias_classes(img)
    print(type(func))
    print(type(np.zeros(img.shape)))
    print(func.shape)
    print(energia_externa(img, calcula_ocorrencias_classes(img)))
