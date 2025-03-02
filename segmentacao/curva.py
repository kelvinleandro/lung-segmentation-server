import numpy as np
from math import atan2, sin, cos, pi

def crisp_inicial(
    imagem: np.ndarray, 
    lim_infY: int, 
    lim_supY: int, 
    lim_infX: int, 
    lim_supX: int
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

    angulos = np.linspace(0, 2 * np.pi, quantidade_pixels+1)
    curva = ponto + np.c_[np.cos(angulos), np.sin(angulos)] * raio

    assert np.all(curva > 0), "Os pontos da curva devem sempre ser maior que 0"

    return curva[:-1].astype(np.int16)

def calcular_angulo(p1:np.ndarray, 
                    p2:np.ndarray, 
                    p3:np.ndarray
) -> float:
    """
    Calcula o ângulo formado pelos três pontos (em graus).
    Args:
        pontos p1(i-1), p2(i) e p3(i+1)
    Return:
        angulo em graus entre os 3 pontos
    
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0) #Manter o cos entre -1 e 1
    
    return np.degrees(np.arccos(cos_theta))


def remover_pontos(
    curva: np.ndarray, 
    alpha: float = 20
) -> np.ndarray:
    """
    Recebe pontos da curva e um ângulo mínimo para remover pontos da curva.

    Args:
        curva (np.darray): Pontos da curva.
        alpha (float): Ângulo mínimo entre dois pontos para remover da curva.
    Returns:
        np.darray: Curva com os pontos removidos.
    """

    assert curva.shape[-1] == 2, "A curva deve ser um array de shape [n,2]"

    curva = curva[np.insert(np.any(np.diff(curva, axis=0), axis=1), 0, True)]
    
    curva_filtrada = [curva[0]]
    
    for i in range(1, len(curva) - 1):
        angulo = calcular_angulo(curva[i-1], curva[i], curva[i+1])
        if angulo > alpha:
            curva_filtrada.append(curva[i])
    
    curva_filtrada.append(curva[-1])
    return np.array(curva_filtrada, dtype=np.int16)


def no_pulmao(ponto, imagem):
    """
    Verifica se um ponto está dentro do pulmão baseado nos valores de UH
    
    Args:
        ponto (np.array): Coordenadas (x, y) do ponto
        imagem (np.ndarray): Imagem do pulmão em UH
        
    Returns:
        bool: True se o ponto está no pulmão, False caso contrário
    """
    x, y = int(round(ponto[0])), int(round(ponto[1]))
    
    # Verificar se o ponto está dentro dos limites da imagem
    if x < 0 or x >= imagem.shape[1] or y < 0 or y >= imagem.shape[0]:
        return False
    
    # Verificar se o valor do pixel está entre -1000 UH e -500 UH (tecido pulmonar)
    valor_uh = imagem[y, x]
    return -1000 <= valor_uh <= -500

def na_curva(ponto, curva):
    """
    Verifica se um ponto está dentro da curva
    
    Args:
        ponto (np.array): Coordenadas (x, y) do ponto
        curva (np.ndarray): Pontos da curva de formato (n, 2)
        
    Returns:
        bool: True se o ponto está dentro da curva, False caso contrário
    """
    # Algoritmo de ray casting para determinar se ponto está dentro da curva
    x, y = ponto
    n = len(curva)
    dentro = False
    
    for i in range(n):
        x1, y1 = curva[i]
        x2, y2 = curva[(i + 1) % n]
        
        # Verificar se o raio cruza a linha
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            dentro = not dentro
            
    return dentro

def adicionar_pontos(curva, imagem, d_max):
    """
    Recebe pontos da curva, a imagem de pulmão em UH e a distância mínima para adicionar pontos na curva

    Args:
        curva (np.ndarray): Pontos da curva.
        imagem (np.ndarray): Imagem do pulmão.
        d_max (float): Distância máxima entre dois pontos (acima disso, adiciona-se um ponto)
    Returns:
        np.ndarray: Curva com os pontos adicionados
    """
    # Inicializa nova curva com pontos existentes
    nova_curva = curva.copy()
    
    # Quantidade de pontos originais na curva
    n = len(curva)
    
    # Inicializa lista de pontos a serem adicionados
    pontos_a_adicionar = []
    
    # Verifica cada par consecutivo de pontos
    for i in range(n):
        # Ponto atual e próximo (o último ponto conecta com o primeiro)
        v1 = curva[i]
        v2 = curva[(i + 1) % n]
        
        # Calcula a distância euclidiana entre os pontos
        distancia = np.sqrt(np.sum((v2 - v1)**2))
        
        # Se a distância for maior que a máxima, adiciona um ponto entre eles
        if distancia > d_max:
            # Calcula o ponto médio entre v1 e v2
            ponto_medio = (v1 + v2) / 2
            
            # Verifica se o ponto médio está no pulmão
            if no_pulmao(ponto_medio, imagem):
                # Se estiver no pulmão, adiciona diretamente
                pontos_a_adicionar.append((i + 1, ponto_medio))
            else:
                # Se não estiver no pulmão, precisamos deslocar o ponto para dentro do pulmão
                
                # Calcula o ângulo da semi-reta formada por v1 e v2
                angulo = atan2(v2[1] - v1[1], v2[0] - v1[0])
                
                # Calcula as direções perpendiculares
                angulo1 = angulo + pi/2
                angulo2 = angulo - pi/2
                
                # Pontos nas duas direções perpendiculares (pequena distância para verificação)
                p1 = np.array([ponto_medio[0] + 5 * cos(angulo1), ponto_medio[1] + 5 * sin(angulo1)])
                p2 = np.array([ponto_medio[0] + 5 * cos(angulo2), ponto_medio[1] + 5 * sin(angulo2)])
                
                # Determina qual direção está dentro da curva
                angulo_correto = angulo1 if na_curva(p1, curva) else angulo2
                
                # Procura pelo primeiro ponto na direção correta que está dentro do pulmão
                encontrou_ponto = False
                for dist in range(1, 51):  # Testa distâncias de 1 a 50
                    ponto_teste = np.array([
                        ponto_medio[0] + dist * cos(angulo_correto), 
                        ponto_medio[1] + dist * sin(angulo_correto)
                    ])
                    
                    if no_pulmao(ponto_teste, imagem):
                        pontos_a_adicionar.append((i + 1, ponto_teste))
                        encontrou_ponto = True
                        break
                
                # Se não encontrou nenhum ponto válido, adiciona o ponto médio mesmo assim
                if not encontrou_ponto:
                    pontos_a_adicionar.append((i + 1, ponto_medio))
    
    # Adiciona os novos pontos à curva, em ordem
    pontos_a_adicionar.sort(key=lambda x: x[0])
    
    # Adiciona os pontos na nova curva
    offset = 0
    for idx, ponto in pontos_a_adicionar:
        nova_curva = np.insert(nova_curva, idx + offset, ponto, axis=0)
        offset += 1
        
    return nova_curva