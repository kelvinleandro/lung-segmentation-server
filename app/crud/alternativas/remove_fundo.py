import cv2
import numpy as np


def remove_fundo(
    mascara: np.ndarray, area_minima: int = 3000, area_maxima: int = 50000
) -> tuple:
    """
    Mantém apenas contornos cujas áreas estão dentro do intervalo especificado e que não tocam a borda da imagem.
    Agora os contornos não podem estar dentro de uma margem de até 10 pixels das bordas.

    Parâmetros:
        mascara (np.ndarray): Máscara binária com os contornos.
        area_minima (int): Área mínima permitida para os contornos (default: 3000).
        area_maxima (int): Área máxima permitida para os contornos (default: 50000).

    Retorna:
        tuple:
            - dict: Dicionário onde cada chave é uma string (e.g., "contorno_0") e o valor é o contorno.
            - dict: Dicionário onde cada chave é uma string (e.g., "contorno_0") e o valor é o contorno válido do pulmão.
    """

    #  margem (int): Distância mínima permitida das bordas  (10 pixels).
    margem: int = 10

    # Encontrar contornos na máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dicionário para armazenar os contornos válidos
    contornos_validos_dict = {}

    # Dimensões da imagem
    altura, largura = mascara.shape

    # Filtrar os contornos que não tocam a borda e estão na faixa de área permitida
    for i, contorno in enumerate(contornos):
        # Verificar se o contorno tem um perímetro válido (> 0)
        if cv2.arcLength(contorno, True) == 0:
            continue  # Ignora contornos inválidos

        # Verifica se algum ponto do contorno está dentro da margem das bordas
        if np.any(
            (
                contorno[:, 0, 0] <= margem
            )  # Ponto a até 'margem' pixels da borda esquerda
            | (
                contorno[:, 0, 0] >= largura - margem
            )  # Ponto a até 'margem' pixels da borda direita
            | (
                contorno[:, 0, 1] <= margem
            )  # Ponto a até 'margem' pixels da borda superior
            | (
                contorno[:, 0, 1] >= altura - margem
            )  # Ponto a até 'margem' pixels da borda inferior
        ):
            continue  # Ignora contornos que tocam a borda ou estão na margem

        # Calcula a área do contorno
        area = cv2.contourArea(contorno)
        if area_minima <= area <= area_maxima:
            # Salva o contorno no dicionário
            contornos_validos_dict[f"contorno_{i}"] = contorno.squeeze().tolist()

    todos_contornos_dict = {
        f"contorno_{i}": contorno.squeeze().tolist()
        for i, contorno in enumerate(contornos)
    }
    return (
        todos_contornos_dict,
        contornos_validos_dict,
    )
