import numpy as np
from carregar import carregar_imagem
from classificacao import calcula_ocorrencias_classes


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
        imagem: Imagem para calcular a energia.
        probabilidade: Probabilidade de ocorrência de classe de todos os pontos.
        probablidade3: Limiar da probabilidade 3 para definir o valor da energia crispy
        como 0.
        probablidade4: Limiar da probabilidade 4 para definir o valor da energia crispy
        como 0.
    Return:
        energia: Matriz com as energias crispy de todos os pontos da imagem.
    """
    energia = np.zeros(imagem.shape)
    for (x, y), value in np.ndenumerate(energia):
        if not (
            probabilidade[2][x][y] < probablidade3
            and probabilidade[3][x][y] <= probablidade4
        ):
            energia[x][y] = S(imagem, x, y)

    return energia


def S(imagem: np.ndarray, x: int, y: int) -> float:
    return 1


if "__main__":
    img = carregar_imagem("data/pulmao2/30.dcm")
    print(type(img))
    func = calcula_ocorrencias_classes(img)
    print(type(func))
    print(type(np.zeros(img.shape)))
    print(func.shape)
    print(energia_externa(img, calcula_ocorrencias_classes(img)))
