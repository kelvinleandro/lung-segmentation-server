import cv2
import numpy as np
from otsu import aplicar_otsu

def dividir_imagem(imagem: np.ndarray, grid_size: tuple, overlap: int) -> list:
    #PDivide a imagem em subimagens
    altura, largura = imagem.shape
    sub_imagens = []

    # Calcula o tamanho de cada sub-região
    passo_y = (altura + overlap * (grid_size[0] - 1)) // grid_size[0]
    passo_x = (largura + overlap * (grid_size[1] - 1)) // grid_size[1]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Define os limites da sub-região
            y_inicio = max(0, i * passo_y - overlap)
            y_fim = min(altura, (i + 1) * passo_y + overlap)
            x_inicio = max(0, j * passo_x - overlap)
            x_fim = min(largura, (j + 1) * passo_x + overlap)

            # Extrai a sub-região
            sub_imagem = imagem[y_inicio:y_fim, x_inicio:x_fim]
            sub_imagens.append(((y_inicio, y_fim, x_inicio, x_fim), sub_imagem))

    return sub_imagens

def combinar_sub_imagens(imagem: np.ndarray, sub_imagens: list) -> np.ndarray:
    #Usa as subimagens para montar uma imagem só de saída
    imagem_final = np.zeros_like(imagem)

    for (y_inicio, y_fim, x_inicio, x_fim), sub_imagem in sub_imagens:
        imagem_final[y_inicio:y_fim, x_inicio:x_fim] = sub_imagem

    return imagem_final

def segmentar_imagem(imagem: np.ndarray, grid_size: tuple = (2, 3), overlap: int = 20) -> np.ndarray:
    # Chama a função de divisão
    sub_imagens = dividir_imagem(imagem, grid_size, overlap)

    # Processa cada sub-região com o método de Otsu
    sub_imagens_processadas = []
    for limites, sub_imagem in sub_imagens:
        sub_imagem_processada = aplicar_otsu(sub_imagem)
        sub_imagens_processadas.append((limites, sub_imagem_processada))

    # Combina as sub-regiões processadas
    imagem_segmentada = combinar_sub_imagens(imagem, sub_imagens_processadas)

    return imagem_segmentada