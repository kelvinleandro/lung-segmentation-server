import numpy as np

def converter_hu_para_cinza(imagem_hu, hu_min=-1000, hu_max=2000) -> np.ndarray:
    """
    Converte Hounsfield Units (HU) para escala de cinza.

    args:
        input_path: np.ndarray - Imagem em Hounsfield Units (HU)
    return:
        np.ndarray - Imagem em escala de cinza
    """

    largura_hu = hu_max - hu_min

    imagem_hu = imagem_hu.astype(np.float32)
    imagem_escala_cinza = np.clip((255 * (imagem_hu - hu_min)) / largura_hu, 0, 255)

    return imagem_escala_cinza.astype(np.uint8)
