def forca_adaptativa(pontos: np.ndarray, indice: int) -> float:
    """
    Recebe os pontos em ordem anti-horária da curva e calcula a força adaptativa
    no ponto de índice 'indice'.

    Args:
        pontos (np.ndarray): Pontos da curva.
        indice (int): Índice do ponto.
    Returns:
        float: Força adaptativa no ponto de índice 'indice'.
    """
    pm = (pontos[(indice - 1)%len(pontos)] + pontos[(indice + 1)%len(pontos)])/2
    v1 = pontos[(indice + 1)%len(pontos)] - pontos[indice]
    v2 = pontos[(indice - 1)%len(pontos)] - pontos[indice]
    vet = np.sign(np.linalg.det([v1, v2]))
    if vet  == 0:
        return 0
    return np.linalg.norm(pm + vet*pontos[indice])