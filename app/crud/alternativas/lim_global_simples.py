import numpy as np
import cv2


def aplicar_lim_global_simples(
    imagem_cinza: np.ndarray, limiar=50, delta_limiar=5
) -> tuple:
    """
    Traça o contorno do pulmão da imagem utilizando Limiarização global simples.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.
        limiar (int): Valor a ser aplicado como limiar inicial na imagem
        delta_limiar (int): Valor usado como critério de parada no ajuste do limiar
    Retorna:
        tuple:
            - np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            - dict: Dicionário onde cada chave é uma string (e.g., "contorno_0")
                    e o valor é o contorno válido.
    """
    # novas imagens são criadas
    imagem_cinza_binario = np.zeros_like(imagem_cinza)
    imagem_cinza_binario = imagem_cinza_binario.astype(np.uint8)

    # Valores de limiares inicializados para o ajuste do limiar
    mod_limiar_dif = float("inf")
    limiar_n_anterior = 0
    limiar_n = limiar

    # Ajuste do limiar
    while mod_limiar_dif > delta_limiar:
        # Máscaras para separar os pixels
        grupo_maior = imagem_cinza[imagem_cinza > limiar_n]
        grupo_menor = imagem_cinza[imagem_cinza <= limiar_n]

        # Cálculo das médias, evitando divisões por zero
        media_maior = np.mean(grupo_maior) if grupo_maior.size > 0 else 0
        media_menor_igual = np.mean(grupo_menor) if grupo_menor.size > 0 else 0

        # Novo limiar é calculado
        limiar_n_anterior = limiar_n
        limiar_n = (media_maior + media_menor_igual) / 2

        # diferença entre os limiares é recalculada
        mod_limiar_dif = abs(limiar_n_anterior - limiar_n)

    # aplicada limiar
    _, imagem_cinza_binario = cv2.threshold(
        imagem_cinza, limiar_n, 255, cv2.THRESH_BINARY
    )

    # inverte a imagem binária
    imagem_cinza_binario_invertida = (255 - imagem_cinza_binario).astype(np.uint8)

    return imagem_cinza_binario_invertida
