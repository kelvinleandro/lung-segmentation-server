import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from crud.alternativas.remove_fundo import remove_fundo

def aplicar_limiarizacao_propriedades(imagem, tamanho_janela=151, a=1.0, b=0.5,
                                       usar_media_global=False,
                                       aplicar_interpolacao=True) -> tuple:
    """
    Aplica limiarização variável baseada na média e no desvio-padrão locais.
    Calcula um limiar para cada ponto da imagem de forma vetorizada.

    args:
        imagem: np.ndarray - Imagem de entrada (em escala de cinza).
        tamanho_janela: int - Define o tamanho da vizinhança usada para
                              cálculo dos limiares.
        a: float - Constante para ajustar a contribuição do desvio padrão.
        b: float - Constante para ajustar a contribuição da média.
        usar_media_global: bool - Se True, usa a média global da imagem.
        aplicar_interpolacao: bool - Se True, aplica interpolação.

    return:
        tuple:
            np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            dict: Dicionário onde cada chave é uma string (e.g., "contorno_0")
                  e o valor é o contorno válido.
    """
    media_local = uniform_filter(imagem.astype(np.float32), tamanho_janela)

    quadrado_local = uniform_filter(imagem.astype(np.float32) ** 2, tamanho_janela)

    desvio_local = np.sqrt(quadrado_local - media_local ** 2)

    media_usada = np.mean(imagem) if usar_media_global else media_local

    Txy = a * desvio_local + (b * media_usada)

    limiarizada = np.where(imagem > Txy, 255, 0).astype(np.uint8)

    if aplicar_interpolacao:
        limiarizada = cv2.resize(limiarizada, None, fx=1.2, fy=1.2,
                                 interpolation=cv2.INTER_CUBIC)
        limiarizada = cv2.resize(limiarizada, (imagem.shape[1], imagem.shape[0]),
                                 interpolation=cv2.INTER_AREA)

    imagem_limiarizada_inv = cv2.bitwise_not(limiarizada)

    return imagem_limiarizada_inv