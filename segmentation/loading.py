import pydicom as dicom
import numpy as np

def load_image(input_path: str) -> np.ndarray:
    ds = dicom.dcmread(input_path)

    assert ds.Modality == "CT", "Only CT images are supported"

    rescale_intercept = getattr(ds, "RescaleIntercept", 0)
    rescale_slope = getattr(ds, "RescaleSlope", 1)

    hu_image = ds.pixel_array * rescale_slope + rescale_intercept
    return hu_image
