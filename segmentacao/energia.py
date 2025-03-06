import numpy as np
import cv2
import numba

from segmentacao.forca import forca_adaptativa, forca_continuidade


def energia_externa(
    imagem: np.ndarray,
    probabilidade: np.ndarray,
    probablidade3: float = 0.2,
    probablidade4: float = 0.15,
) -> np.ndarray:
    """
    Recebe a imagem e as probabilidades de ocorrência de classe da imagem e retorna
    a matriz com energia externa crisp de cada ponto imagem. Pode receber os limiares
    das probabilidade P(3) e P(4) para experimentação posterior.

    Args:
        imagem (np.ndarray): Imagem para calcular a energia.
        probabilidade (np.ndarray): Probabilidade de ocorrência de classe de todos os
        pontos.
        probablidade3 (float): Limiar da probabilidade 3 para definir o valor da energia
        crispy como 0.
        probablidade4 (float): Limiar da probabilidade 4 para definir o valor da energia
        crispy como 0.
    Return:
        energia (np.ndarray): Matriz das energias crispy de todos os pontos da imagem.
    """
    # Cálculo do Sobel
    sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
    energia = np.sqrt(sobel_x**2 + sobel_y**2)

    # Mascara de probabilidade
    mask = (probabilidade[2] >= probablidade3) | (probabilidade[3] > probablidade4)
    energia[~mask] = 0

    return energia


def energia_interna_adaptativa(
    curva: np.ndarray, indice: int, w_adapt: float = 0.1, w_cont: float = 0.6
) -> np.floating:
    """
    Calcula a energia interna adaptativa, utilizando a força adaptativa e força
    de continuidade ponderadas

    Args:
        curva (np.ndarray): Curva de n pontos
        indice (int): Indice do ponto da curva
        w_adapt (float): Peso da energia interna adaptativa
        w_cont (float): Peso da energia de continuidade
    Return:
        energia_interna_adaptativa (float): Energia interna adaptativa do ponto da curva
    """
    adaptativa = w_adapt * forca_adaptativa(curva, indice)
    continuidade = w_cont * forca_continuidade(curva, indice)
    return adaptativa + continuidade


def energia_total(
    curva: np.ndarray,
    indice: int,
    energia_crisp: np.ndarray,
    w_adapt: float = 0.1,
    w_cont: float = 0.6,
) -> np.floating:
    """
    Calcula a energia total, somando a energia interna adaptativa e a energia
    crisp/externa

    Args:
        curva (np.ndarray): Curva de n pontos
        indice (int): Indice do ponto da curva
        energia_crisp (np.ndarray): Matriz de energia externa
        w_adapt (float): Peso da energia interna adaptativa
        w_cont (float): Peso da energia de continuidade
    Return:
        energia_total (float): Energia total do ponto da curva
    """
    x, y = curva[indice]
    return np.float64(
        energia_interna_adaptativa(curva, indice, w_adapt, w_cont) + energia_crisp[y, x]
    )


def minimiza_energia(
    curva: np.ndarray,
    indice: int,
    energia_crisp: np.ndarray,
    area_de_busca: int = 9,
    w_cont=0.6,
    w_adapt=0.1,
) -> np.ndarray:
    """
    Minimiza a energia para um ponto de uma dada curva na área de busca
    especificada e retorna o ponto com energia mínima.

    Args:
        curva (np.ndarray): Curva de n pontos
        indice (int): Indice do ponto da curva
        energia_crisp (np.ndarray): Matriz de energia externa
        area_de_busca (int): Tamanho da área de busca
        w_cont (float): Peso da energia de continuidade
        w_adapt (float): Peso da energia interna adaptativa

    Return:
        melhor_ponto (np.ndarray): Melhor ponto encontrado na área de busca

    """
    assert area_de_busca % 2 == 1 and area_de_busca > 0, "Area de busca deve ser impar"

    amplitude_de_indices = np.arange(-(area_de_busca // 2), area_de_busca // 2 + 1)
    deslocamentos = np.array(
        [[dx, dy] for dx in amplitude_de_indices for dy in amplitude_de_indices]
    )
    possiveis_pontos = deslocamentos + curva[indice]

    energias_calculadas = np.zeros(area_de_busca**2)

    for i in numba.prange(area_de_busca**2):
        curva_modificada = curva.copy()
        ponto_candidato = possiveis_pontos[i]
        curva_modificada[indice] = ponto_candidato
        energia = energia_total(
            curva_modificada,
            numba.int32(indice),
            energia_crisp,
            w_adapt=w_adapt,
            w_cont=w_cont,
        )

        energias_calculadas[i] = energia

    melhor_energia_indice = np.argmax(energias_calculadas)

    return possiveis_pontos[melhor_energia_indice]
