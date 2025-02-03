import numpy as np
import cv2
from cv2.typing import MatLike

U_CLASSES = np.array(
    [
        (-1_000, -950),
        (-950, -500),
        (-500, -100),
        (-100, 100),
        (600, 2_000),
    ]
)


def calcula_ocorrencias_classes(src: MatLike) -> np.ndarray:
    """
    Calcula a quantidade de ocorrências de cada classe para cada pixel usando
    operações vetorizadas.

    Args:
        src (np.ndarray): Imagem de entrada(DICOM).

    Returns:
        np.ndarray: Matriz tridimensional (5, h, w) com as ocorrências das classes.
    """
    # Adiciona borda refletida para lidar com pixels nas extremidades
    pad_src = cv2.copyMakeBorder(src, 4, 4, 4, 4, cv2.BORDER_REFLECT)
    h, w = src.shape
    ocorrencias = np.zeros((5, h, w), dtype=np.uint8)

    for i in range(5):
        lower, upper = U_CLASSES[i]
        # Cria máscara para a classe i (upper é exclusivo, subtrai 1)
        mask = cv2.inRange(pad_src, np.array(lower), np.array(upper - 1))
        # Converte para 0s e 1s e calcula a soma 9x9
        ocorrencias[i] = cv2.boxFilter(mask // 255, -1, (9, 9), normalize=False)[
            4:-4, 4:-4
        ]

    return ocorrencias


def probabilidade_classes(ocorrencias: MatLike) -> np.ndarray:
    """
    Calcula as probabilidades das classes usando operações vetorizadas.

    Args:
        ocorrencias (np.ndarray): Matriz de ocorrências (5, h, w).

    Returns:
        np.ndarray: Matriz de probabilidades (5, h, w) com dtype float32.
    """
    soma = np.squeeze(ocorrencias.sum(axis=0, keepdims=True))  # Soma total por pixel
    mask = soma == 0  # Máscara para divisão por zero

    # Calcula probabilidades e trata divisão por zero
    probabilidades = np.zeros_like(ocorrencias, dtype=np.float32)
    np.divide(ocorrencias, soma, out=probabilidades, where=~mask)
    probabilidades[:, mask] = 1.0 / 5  # Comportamento original (possível erro lógico)

    return probabilidades
