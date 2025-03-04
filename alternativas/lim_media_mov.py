import cv2
import numpy as np
from segmentacao.remove_fundo import remove_fundo

def aplicar_limiarizacao_media_movel(imagem, n=171, b=0.8,
                                     aplicar_interpolacao=True) -> tuple:
    """
    Recebe uma imagem em escala de cinza, aplica limiarização usando
    média móvel e remove o fundo da imagem.

    args:
        imagem: np.ndarray - Imagem de entrada (em escala de cinza).
        n: int - Número de pontos para a média móvel.
        b: float - Fator de ajuste do limiar.
        aplicar_interpolacao: bool - Se True, aplica o desfoque mediano.

    return:
        tuple:
            np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            dict: Dicionário onde cada chave é uma string (e.g., "contorno_0")
                  e o valor é o contorno válido.
    """

    altura, largura = imagem.shape
    limiarizada = np.zeros_like(imagem, dtype=np.uint8)

    for y in range(altura):
        linha = imagem[y, :].astype(np.float32)
        media_movel = np.zeros_like(linha)

        media_movel[0] = linha[0] / n
        soma = linha[0]

        for x in range(1, largura):
            if x < n:
                soma += linha[x]
                media_movel[x] = soma / (x + 1)
            else:
                soma += linha[x] - linha[x - n]
                media_movel[x] = soma / n

        # Aplicar o limiar adaptativo
        limiar = b * media_movel
        limiarizada[y, :] = np.where(linha > limiar, 255, 0)

    if aplicar_interpolacao:
    # Refinar a máscara com interpolação
        limiarizada = cv2.resize(limiarizada, None, fx=1.2, fy=1.2,
                                    interpolation=cv2.INTER_CUBIC)
        limiarizada = cv2.resize(limiarizada, (imagem.shape[1], imagem.shape[0]),
                                    interpolation=cv2.INTER_AREA)

    imagem_limiarizada_inv = cv2.bitwise_not(limiarizada)

    return remove_fundo(imagem_limiarizada_inv)
