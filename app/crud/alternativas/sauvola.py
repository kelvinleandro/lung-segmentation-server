import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from crud.alternativas.remove_fundo import remove_fundo

def aplicar_sauvola(imagem, tamanho_janela=171, k=0.02,
                                  aplicar_interpolacao=True,
                                  aplicar_morfologia=True,
                                  tamanho_kernel=3,
                                  iteracoes_morfologia=1) -> tuple:
    """
    Aplica método de Sauvola para segmentação adaptativa. É uma técnica
    de limiarização local útil para imagens que não são uniformes, especialmente
    para reconhecimento de texto. Ao invés de calcular um único limiar global
    para a imagem inteira, vários limiares são calculados para todos os pixels
    utilizando fórmulas específicas que levam em conta a média e o desvio-padrão
    para cada vizinhança local.

    args:
        imagem: np.ndarray - Imagem de entrada (em escala de cinza).
        tamanho_janela: int - Tamanho da janela para cálculo do limiar.
        k: float - Parâmetro de ajuste do método de Sauvola.
        aplicar_interpolacao: bool - Se True, aplica interpolação para
                                     suavizar a máscara.
        aplicar_morfologia: bool - Se True, aplica operações morfológicas
                                     para refinamento.
        tamanho_kernel: int - Tamanho do kernel para operações morfológicas.
        iteracoes_morfologia: int - Número de iterações para
                                    operações morfológicas.

    return:
        tuple:
            np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            dict: Dicionário onde cada chave é uma string (e.g., "contorno_0")
                  e o valor é o contorno válido.
    """
    imagem_suavizada = cv2.medianBlur(imagem, 5)
    thresh_sauvola = threshold_sauvola(imagem_suavizada,
                                       window_size=tamanho_janela, k=k)
    imagem_segmentada = (imagem_suavizada > thresh_sauvola).astype(np.uint8) * 255

    if aplicar_morfologia:
        kernel = np.ones((tamanho_kernel, tamanho_kernel), np.uint8)
        imagem_segmentada = cv2.morphologyEx(imagem_segmentada, cv2.MORPH_CLOSE,
                                             kernel,
                                             iterations=iteracoes_morfologia)
        imagem_segmentada = cv2.morphologyEx(imagem_segmentada, cv2.MORPH_OPEN,
                                             kernel,
                                             iterations=iteracoes_morfologia)

    if aplicar_interpolacao:
        imagem_segmentada = cv2.resize(imagem_segmentada, None, fx=1.2, fy=1.2,
                                       interpolation=cv2.INTER_CUBIC)
        imagem_segmentada = cv2.resize(imagem_segmentada, (imagem.shape[1],
                                                           imagem.shape[0]),
                                       interpolation=cv2.INTER_AREA)

    imagem_final = cv2.bitwise_not(imagem_segmentada)

    return imagem_final