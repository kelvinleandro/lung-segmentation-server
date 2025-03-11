import pydicom as dicom
import numpy as np


def carregar_imagem(input_path: str) -> np.ndarray:
    """
    Carrega imagem do tipo DICOM e retorna a imagem em Hounsfield Units (HU).

    args:
        input_path: str - Caminho para a imagem DICOM
    return:
        np.ndarray - Imagem em Hounsfield Units
    """
    ds = dicom.dcmread(input_path)

    assert ds.Modality == "CT", "Somente imagens do tipo CT são suportadas"

    rescale_intercept = getattr(ds, "RescaleIntercept", 0)
    rescale_slope = getattr(ds, "RescaleSlope", 1)

    hu_image = ds.pixel_array * rescale_slope + rescale_intercept
    return hu_image
