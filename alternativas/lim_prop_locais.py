import cv2
import numpy as np
from segmentacao.remove_fundo import remove_fundo

def aplicar_limiarizacao_propriedades(imagem, tamanho_janela=151, a=1.0, b=0.5, 
                                      usar_media_global=False, 
                                      aplicar_interpolacao=True):
    """
    Aplica limiarização variável baseada na média e no desvio padrão locais.
    tamanho_janela: Define o tamanho da vizinhança usada para cálculo dos limiares.
    a, b: Constantes para ajustar a contribuição do desvio padrão e da média.
    usar_media_global: Se True, usa a média global da imagem ao invés da local.
    """
    altura, largura = imagem.shape
    limiarizada = np.zeros_like(imagem, dtype=np.uint8)
    
    # Calcula média global, se necessário
    media_global = np.mean(imagem) if usar_media_global else None
    
    # Percorre a imagem aplicando limiarização local
    for y in range(altura):
        for x in range(largura):
            y1, y2 = max(0, y - tamanho_janela // 2), min(altura, y + tamanho_janela // 2)
            x1, x2 = max(0, x - tamanho_janela // 2), min(largura, x + tamanho_janela // 2)
            
            regiao = imagem[y1:y2, x1:x2]
            media_local = np.mean(regiao)
            desvio_local = np.std(regiao)
            
            media_usada = media_global if usar_media_global else media_local

            Txy = a * desvio_local + (b * media_usada)
            limiarizada[y, x] = 255 if imagem[y, x] > Txy else 0
    
    if aplicar_interpolacao:
        limiarizada = cv2.resize(limiarizada, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        limiarizada = cv2.resize(limiarizada, (limiarizada.shape[1], limiarizada.shape[0]), interpolation=cv2.INTER_AREA)

    imagem_limiarizada_inv = cv2.bitwise_not(limiarizada)

    return remove_fundo(imagem_limiarizada_inv)