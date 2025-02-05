import numpy as np
from carregar import carregar_imagem
from classificacao import calcula_ocorrencias_classes

def energia_externa(imagem: np.ndarray, probabilidade: np.ndarray, probablidade3: float = 0.2, probablidade4: float = 0.15) -> np.ndarray:
    energia = np.zeros(imagem.shape)
    for (x,y), value in np.ndenumerate(energia):
        if(not (probabilidade[2][x][y]<probablidade3 and probabilidade[3][x][y]<=probablidade4)):
            energia[x][y] = S(imagem,x,y)
    
    return energia


def S(imagem: np.ndarray, x:int, y:int) -> float: 
  return 1

if "__main__":
    img = carregar_imagem('data/pulmao2/30.dcm')
    print(type(img))
    func = calcula_ocorrencias_classes(img)
    print(type(func))
    print(type(np.zeros(img.shape)))
    print(func.shape)
    print(energia_externa(img,calcula_ocorrencias_classes(img)))
