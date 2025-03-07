import numpy as np
from segmentacao import lim_global_simples

def test_lim_global_simples():
    img = np.array([[0, 255], [128, 192]], dtype=np.uint8)
    limiar = 128
    img_limiarizada = lim_global_simples(img, limiar)
    assert np.array_equal(img_limiarizada, np.array([[0, 255], [0, 255]], dtype=np.uint8))
