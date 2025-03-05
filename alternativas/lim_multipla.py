import cv2
import numpy as np
from segmentacao.remove_fundo import remove_fundo
from segmentacao.carregar import carregar_imagem
import alternativas.hu_para_cinza as hu
import matplotlib.pyplot as plt

def classificar_pixel(
    pixel_value: int,
    lim_hiperaeradas: tuple = (0, 8),
    lim_normalmente_aeradas: tuple = (8, 42),
    lim_pouco_aeradas: tuple = (42, 76),
    lim_nao_aeradas: tuple = (76, 93),
    lim_osso: tuple = (136, 255),
) -> int:
    """
    Classifica o valor de cada pixel com base no intervalo de Hounsfield Units (HU) convertido para escala de cinza.

    Parâmetros:
        pixel_value (int): Valor do pixel em HU (intensidade da imagem).
        lim_hiperaeradas (tuple): Intervalo para hiperaeradas (default: (0, 8)).
        lim_normalmente_aeradas (tuple): Intervalo para normalmente aeradas (default: (8, 42)).
        lim_pouco_aeradas (tuple): Intervalo para pouco aeradas (default: (42, 76)).
        lim_nao_aeradas (tuple): Intervalo para não aeradas (default: (76, 93)).
        lim_osso (tuple): Intervalo para osso (default: (136, 255)).

    Retorna:
        int: Classe do pixel (0 a 5).
    """

    # Classificação dos intervalos de escala de cinza
    if lim_hiperaeradas[0] <= pixel_value < lim_hiperaeradas[1]:
        return 0  # u0 - hiperaeradas
    elif lim_normalmente_aeradas[0] <= pixel_value < lim_normalmente_aeradas[1]:
        return 1  # u1 - normalmente aeradas
    elif lim_pouco_aeradas[0] <= pixel_value < lim_pouco_aeradas[1]:
        return 2  # u2 - pouco aeradas
    elif lim_nao_aeradas[0] <= pixel_value < lim_nao_aeradas[1]:
        return 3  # u3 - não aeradas
    elif lim_osso[0] <= pixel_value <= lim_osso[1]:
        return 4  # u4 - osso
    else:
        return 5  # u5 - áreas não classificadas


def limiarizacao_multipla(
    imagem_cinza: np.ndarray,
    lim_hiperaeradas: tuple = (0, 8),
    lim_normalmente_aeradas: tuple = (8, 42),
    lim_pouco_aeradas: tuple = (42, 76),
    lim_nao_aeradas: tuple = (76, 93),
    lim_osso: tuple = (136, 255),
    ativacao_hiperaeradas: bool = True,
    ativacao_normalmente_aeradas: bool = True,
    ativacao_pouco_aeradas: bool = True,
    ativacao_nao_aeradas: bool = False,
    ativacao_osso: bool = False,
    ativacao_nao_classificado: bool = False,
) -> np.ndarray:
    """
    Aplica limiarização múltipla para segmentar as áreas de interesse na imagem.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.
        lim_hiperaeradas (tuple): Intervalo para hiperaeradas (default: (0, 8)).
        lim_normalmente_aeradas (tuple): Intervalo para normalmente aeradas (default: (8, 42)).
        lim_pouco_aeradas (tuple): Intervalo para pouco aeradas (default: (42, 76)).
        lim_nao_aeradas (tuple): Intervalo para não aeradas (default: (76, 93)).
        lim_osso (tuple): Intervalo para osso (default: (136, 255)).
        ativacao_*: Variável booleana que coloca no pixel 255 caso true e 0 caso false.


    Retorna:
        np.ndarray: Imagem com classificação dos pixels com base nos intervalos de HU.
            
    """

    # Criar uma imagem para armazenar as classes
    imagem_classes = np.zeros(imagem_cinza.shape, dtype=np.uint8)

    # Percorrer todos os pixels e aplicar a classificação
    for i in range(imagem_cinza.shape[0]):
        for j in range(imagem_cinza.shape[1]):
            pixel_value = int(imagem_cinza[i, j])  # Valor do pixel
            imagem_classes[i, j] = classificar_pixel(
                pixel_value,
                lim_hiperaeradas,
                lim_normalmente_aeradas,
                lim_pouco_aeradas,
                lim_nao_aeradas,
                lim_osso,
            )

    # Gerar a máscara binária com base nas áreas de interesse (u0 a u4)
    mascara_pulmao = np.zeros(imagem_cinza.shape, dtype=np.uint8)

    # Aplicar os limiares com base nos parâmetros fornecidos
    if ativacao_hiperaeradas:
        mascara_pulmao[
            (imagem_cinza >= lim_hiperaeradas[0]) & (imagem_cinza < lim_hiperaeradas[1])
        ] = 255

    if ativacao_normalmente_aeradas:
        mascara_pulmao[
            (imagem_cinza >= lim_normalmente_aeradas[0])
            & (imagem_cinza < lim_normalmente_aeradas[1])
        ] = 255

    if ativacao_pouco_aeradas:
        mascara_pulmao[
            (imagem_cinza >= lim_pouco_aeradas[0])
            & (imagem_cinza < lim_pouco_aeradas[1])
        ] = 255

    if ativacao_nao_aeradas:
        mascara_pulmao[
            (imagem_cinza >= lim_nao_aeradas[0]) & (imagem_cinza < lim_nao_aeradas[1])
        ] = 255

    if ativacao_osso:
        mascara_pulmao[
            (imagem_cinza >= lim_osso[0]) & (imagem_cinza <= lim_osso[1])
        ] = 255

    # Pixels não classificados recebem ativação personalizada
    if ativacao_nao_classificado:
        mascara_pulmao[mascara_pulmao == 0] = 255

    return mascara_pulmao
