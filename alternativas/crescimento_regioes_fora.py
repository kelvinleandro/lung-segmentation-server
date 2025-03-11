import cv2
import numpy as np
from segmentacao.remove_fundo import remove_fundo

def crescimento_regioes_fora(imagem_original):

    hu_min=-1000
    hu_max=2000
    largura_hu = hu_max - hu_min
    imagem_original = imagem_original.astype(np.float32)
    imagem_original = (imagem_original * largura_hu) / 255.0 + hu_min

    """
    Aplica o algoritmo de crescimento de regiões com a semente fora do pulmão.

    Parâmetros:
        imagem (np.ndarray): Imagem de entrada em escala de cinza.
    Retorna:
            - Imagem original com os contornos dos pulmões destacados em vermelho.
            - Imagem com apenas os contornos dos pulmões em vermelho sobre fundo preto.

    Resumo da teoria:
        A técnica de segmentação por crescimento de regiões consiste em agrupar regiões 
        que compartilham propriedades parecidas, como intensidade de pixels, textura ou 
        cor. O método é iniciado a partir de um ou mais pixels denominados de “sementes”, 
        que podem pertencer ou não ao elemento que se deseja segmentar. São definidos 
        critérios de similaridade, que indicam quais pixels devem ser incluídos na região 
        baseados em sua textura, intensidade ou cor. 

	    A partir das sementes e dos critérios de similaridade é iniciado o crescimento da 
        região, onde os pixels que atendem aos critérios são adicionados a região, parando 
        quando não há mais pixels disponíveis a serem adicionados ou o tamanho máximo da 
        região foi atingido.  
        """

    # Adicionar colunas extras (-1000 HU) nas laterais da imagem
    colunas_extras = np.full((imagem_original.shape[0], 10), -1000, dtype=np.int16)
    imagem = np.hstack((colunas_extras, imagem_original, colunas_extras))

    # Janelamento de Intensidade HU (-1000 a -200)
    imagem_janelada = np.clip(imagem.astype(np.float32), -1000, -200)

    # Normalização para 8 bits (0 a 255)
    imagem_normalizada = ((imagem_janelada + 1000) / 800) * 255.0
    imagem_normalizada = np.clip(imagem_normalizada, 0, 255).astype(np.uint8)

    # Criar máscara para identificar o corpo (regiões HU < -968)
    mascara_fundo = (imagem_janelada < -968).astype(np.uint8)
    coordenadas_fundo = np.column_stack(np.where(mascara_fundo == 1))

    # Ajuste dinâmico da tolerância baseado no desvio padrão do fundo
    if len(coordenadas_fundo) > 0:
        valores_fundo = imagem_janelada[mascara_fundo == 1]
        desvio_padrao = np.std(valores_fundo)
        tolerancia = int(desvio_padrao * 2)
    else:
        tolerancia = 50 

    # Restringe a tolerância a um intervalo razoável (10 a 100)
    tolerancia = max(10, min(tolerancia, 100))

    # Criar máscara para o algoritmo de crescimento de regiões
    mascara = np.zeros((imagem_normalizada.shape[0] + 2, imagem_normalizada.shape[1] + 2), np.uint8)

    # Aplicar crescimento de regiões no corpo (semente dentro do corpo)
    ponto_semente = (37, 268)
    cv2.floodFill(imagem_normalizada, mascara, ponto_semente, 255, loDiff=tolerancia, upDiff=tolerancia, flags=8)

    # **Repetir o processo para segmentar o fundo**
    mascara_fundo = (imagem_janelada < -950).astype(np.uint8)
    coordenadas_fundo = np.column_stack(np.where(mascara_fundo == 1))

    if len(coordenadas_fundo) > 0:
        valores_fundo = imagem_janelada[mascara_fundo == 1]
        desvio_padrao = np.std(valores_fundo)
        tolerancia = int(desvio_padrao * 2)
    else:
        tolerancia = 50  # Caso não tenha fundo detectado

    tolerancia = max(10, min(tolerancia, 100))

    # Aplicar crescimento de regiões no fundo (semente no canto da imagem)
    ponto_semente = (1, 1)
    cv2.floodFill(imagem_normalizada, mascara, ponto_semente, 255, loDiff=tolerancia, upDiff=tolerancia, flags=8)

    # Remover as bordas adicionadas na máscara
    mascara = mascara[1:-1, 1:-1]

    # Criar imagem segmentada com os valores HU ajustados
    imagem_segmentada_8bits = np.where(mascara == 1, 255, imagem_normalizada)

    # Inverter a imagem segmentada para possibilitar o processamento posterior
    imagem_segmentada_8bits_invertida = cv2.bitwise_not(imagem_segmentada_8bits)

    # Aplicar a função de remoção de fundo e obter os contornos do pulmão
    pulmao_contornado, contornos_validos = remove_fundo(imagem_segmentada_8bits_invertida)

    return pulmao_contornado, contornos_validos

