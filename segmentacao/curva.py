import numpy as np


def inicializa_curva(
    ponto: np.ndarray,
    raio: int = 30,
    quantidade_pixels: int = 30,
) -> np.ndarray:
    """
    Recebe o ponto inicial do eixo x e y, calcula a curva e retorna a duas
    listas de pontos que representam a curva.

    Args:
        ponto: (x,y) que representa o centro da curva
        raio: raio da curva
        quantidade_pixels: quantidade de pontos que a curva terá
    Return:
        curva: lista de pontos que representam a curva
    Raises:
        AssertionError: caso curva seja menor que 0 em algum ponto
        AssertionError: caso a quantidade_pixels seja menor que 2
    """
    assert quantidade_pixels >= 2, "Quantidade de pixels deve ser sempre maior que 2"

    angulos = np.linspace(0, 2 * np.pi, quantidade_pixels)
    curva = ponto + np.c_[np.cos(angulos), np.sin(angulos)] * raio

    assert np.all(curva > 0), "Os pontos da curva devem sempre ser maior que 0"

    return curva.astype(np.int16)

  
def crisp_inicial(
    imagem: np.ndarray, lim_infY: int, lim_supY: int, lim_infX: int, lim_supX: int
) -> np.ndarray:
    """
    Recebe uma imagem em formato de array NumPy e recorta a região definida pelos
    limites superiores e inferiores (tanto em X quanto em Y). A função identifica
    a posição onde há maior concentração de pixels dentro da faixa de intensidade
    [-1000, -500] e retorna as coordenadas correspondentes.

    Args:
        imagem (np.ndarray): Matriz da imagem de entrada.
        lim_infY (int): Limite inferior do recorte na direção vertical (eixo Y).
        lim_supY (int): Limite superior do recorte na direção vertical (eixo Y).
        lim_infX (int): Limite inferior do recorte na direção horizontal (eixo X).
        lim_supX (int): Limite superior do recorte na direção horizontal (eixo X).

    Returns:
        np.ndarray: Coordenadas (x, y) do ponto com maior concentração de pixels
        dentro da faixa de interesse.

    Raises:
        ValueError: Se os limites fornecidos forem inválidos, ou seja, se
        lim_supY < lim_infY ou lim_supX < lim_infX.

    Example:
        >>> from carregar import carregar_imagem
        >>> imagem = carregar_imagem("../data/pulmao2/60.dcm")
        >>> centro_Esq = crisp_inicial(
        ...     imagem=imagem, lim_infY=180, lim_supY=360, lim_infX=0, lim_supX=255
        ... )
        array([172, 266])
        >>> centro_dir = crisp_inicial(
        ...     imagem=imagem,
        ...     lim_infY=180,
        ...     lim_supY=360,
        ...     lim_infX=256,
        ...     lim_supX=512,
        ... )
        array([335, 289])
    """
    if lim_supY < lim_infY or lim_supX < lim_infX:
        raise ValueError("Os limites fornecidos para X e Y são inválidos.")

    # Recorta a Imagem nos Limites Fornecidos
    imagem = imagem[lim_infY : lim_supY + 1, lim_infX : lim_supX + 1]
    P = np.where((imagem > -1000) & (imagem < -500), 1, 0)

    # Soma os pixels ao longo dos eixos
    X = np.sum(P, axis=1)  # Soma na direção das colunas
    Y = np.sum(P, axis=0)  # Soma na direção das linhas

    return np.array([np.argmax(Y) + lim_infX, np.argmax(X) + lim_infY])
