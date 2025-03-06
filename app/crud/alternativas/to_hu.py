import numpy as np
import pydicom

def converte_para_hu(pixel_array: np.ndarray, ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Converte uma matriz de imagem DICOM para Hounsfield Units (HU).

    Args:
        pixel_array: np.ndarray - Matriz da imagem já extraída do DICOM.
        ds: pydicom.dataset.FileDataset - Objeto DICOM contendo os metadados.

    Returns:
        np.ndarray - Imagem convertida para Hounsfield Units (HU).
    """

    rescale_intercept = getattr(ds, "RescaleIntercept", 0)
    rescale_slope = getattr(ds, "RescaleSlope", 1)

    hu_image = pixel_array * rescale_slope + rescale_intercept
    return hu_image
