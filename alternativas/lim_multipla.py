import cv2
import numpy as np
from segmentacao.remove_fundo import remove_fundo
from segmentacao.carregar import carregar_imagem
import alternativas.hu_para_cinza as hu
import matplotlib.pyplot as plt


def classificar_pixel(pixel_value: int) -> int:
    """
    Classifica o valor de cada pixel com base no intervalo de Hounsfield Units (HU) convertido para escala de cinza.

    Parâmetros:
        pixel_value (int): Valor do pixel em HU (intensidade da imagem).

    Retorna:
        int: Classe do pixel (0 a 5).
    """

    # Limites de HU em escala de cinza
    lim_hiperaeradas_inferior = 0  # Equivale a -1000 HU
    lim_hiperaeradas_superior = 8  # Equavele a -900  HU
    lim_normalmente_aeradas_inferior = 8  # Equivale a -900  HU
    lim_normalmente_aeradas_superior = 42  # Equivale a -500  HU
    lim_pouco_aeradas_inferior = 42  # Equivale a -500  HU
    lim_pouco_aeradas_superior = 76  # Equivale a -100  HU
    lim_nao_aeradas_inferior = 76  # Equivale a -100  HU
    lim_nao_aeradas_superior = 93  # Equivale a  100  HU
    lim_osso_inferior = 136  # Equivale a  600  HU
    lim_osso_superior = 255  # Equivale a  2000 HU

    # Classificação dos intervalos de escala de cinza
    if lim_hiperaeradas_inferior <= pixel_value < lim_hiperaeradas_superior:
        return 0  # u0 - hiperaeradas
    elif (
        lim_normalmente_aeradas_inferior
        <= pixel_value
        < lim_normalmente_aeradas_superior
    ):
        return 1  # u1 - normalmente aeradas
    elif lim_pouco_aeradas_inferior <= pixel_value < lim_pouco_aeradas_superior:
        return 2  # u2 - pouco aeradas
    elif lim_nao_aeradas_inferior <= pixel_value < lim_nao_aeradas_superior:
        return 3  # u3 - não aeradas
    elif lim_osso_inferior <= pixel_value <= lim_osso_superior:
        return 4  # u4 - osso
    else:
        return 5  # u5 - áreas não classificadas


def limiarizacao_multipla(imagem_cinza: np.ndarray) -> tuple:
    """
    Aplica limiarização múltipla para segmentar as áreas de interesse na imagem.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.

    Retorna:
        tuple:
            np.ndarray: Imagem com classificação dos pixels com base nos intervalos de UH.
            dictionary: Dicionário com os contornos válidos relativos ao pulmão.

    """
    # Criar uma imagem para armazenar as classes
    imagem_classes = np.zeros(imagem_cinza.shape, dtype=np.uint8)

    # Percorrer todos os pixels e aplicar a classificação
    for i in range(imagem_cinza.shape[0]):
        for j in range(imagem_cinza.shape[1]):
            pixel_value = int(imagem_cinza[i, j])  # Valor do pixel
            imagem_classes[i, j] = classificar_pixel(pixel_value)

    # Gerar a máscara binária com base nas áreas de interesse (u0 a u4)
    mascara_pulmao = np.zeros(imagem_cinza.shape, dtype=np.uint8)

    # Definindo áreas de interesse (u0 a u4)
    mascara_pulmao[imagem_classes == 0] = 255  # Hiperaeradas
    mascara_pulmao[imagem_classes == 1] = 255  # Normalmente aeradas
    mascara_pulmao[imagem_classes == 2] = 255  # Pouco aeradas
    mascara_pulmao[imagem_classes == 3] = 0  # Não aeradas
    mascara_pulmao[imagem_classes == 4] = 0  # Osso
    mascara_pulmao[imagem_classes == 5] = 0  # Não classificada

    return remove_fundo(mascara_pulmao)
