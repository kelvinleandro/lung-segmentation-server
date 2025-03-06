import cv2
import numpy as np
from crud.alternativas.remove_fundo import remove_fundo

def aplicar_watershed(imagem: np.ndarray, limiar=60,
                      aplicar_interpolacao=True,
                      aplicar_morfologia=True,
                      tamanho_kernel=3,
                      iteracoes_morfologia=2,
                      iteracoes_dilatacao=1,
                      fator_dist_transform=0.3) -> np.ndarray:
    """
    Aplica o algoritmo Watershed para segmentação.

    args:
        imagem (np.ndarray): Imagem de entrada em escala de cinza.
        limiar (float): Valor de limiar para segmentação binária.
                        Pixels abaixo deste valor tornam-se brancos (255),
                        e os acima tornam-se pretos (0).
        aplicar_interpolacao (bool): Se True, aplica interpolação
                                     para refinar a segmentação.
        aplicar_morfologia (bool): Se True, aplica operações morfológicas
                                   para suavizar a segmentação.
        tamanho_kernel (int): Tamanho do kernel utilizado nas operações morfológicas.
        iteracoes_morfologia (int): Número de iterações das operações morfológicas.
        iteracoes_dilatacao (int): Número de iterações para a operação de dilatação.
        fator_dist_transform (float): Fator multiplicador para o limiar da
                                      transformada de distância.

    return:
        tuple:
            np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            dict: Dicionário onde cada chave é uma string (e.g., "contorno_0")
                  e o valor é o contorno válido.
    """

    # Aplicar limiarização binária inversa para destacar os pulmões
    _, mascara_pulmao = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((tamanho_kernel, tamanho_kernel), np.uint8)

    if aplicar_morfologia:
        mascara_pulmao = cv2.morphologyEx(mascara_pulmao, cv2.MORPH_CLOSE, kernel,
                                          iterations=iteracoes_morfologia)
        mascara_pulmao = cv2.morphologyEx(mascara_pulmao, cv2.MORPH_OPEN, kernel,
                                          iterations=iteracoes_morfologia)

    # Criar marcadores para Watershed
    certamente_fundo = cv2.dilate(mascara_pulmao, kernel,
                                  iterations=iteracoes_dilatacao)
    dist_transform = cv2.distanceTransform(mascara_pulmao, cv2.DIST_L2, 5)
    _, certamente_primeiro_plano = cv2.threshold(dist_transform,
                                            fator_dist_transform * dist_transform.max(),
                                            255, 0)

    certamente_primeiro_plano = np.uint8(certamente_primeiro_plano)
    incerto = cv2.subtract(certamente_fundo, certamente_primeiro_plano)

    _, marcadores = cv2.connectedComponents(certamente_primeiro_plano)
    marcadores = marcadores + 1  # Evita regiões de fundo com label 0
    marcadores[incerto == 255] = 0  # Define regiões incertas como 0

    imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    marcadores = cv2.watershed(imagem_bgr, marcadores)

    mascara_segmentada = np.zeros_like(imagem, dtype=np.uint8)
    mascara_segmentada[marcadores > 1] = 255

    if aplicar_interpolacao:
        mascara_segmentada = cv2.resize(mascara_segmentada, None, fx=1.2, fy=1.2,
                                        interpolation=cv2.INTER_CUBIC)
        mascara_segmentada = cv2.resize(mascara_segmentada, (imagem.shape[1],
                                                              imagem.shape[0]),
                                        interpolation=cv2.INTER_AREA)

    return remove_fundo(mascara_segmentada)