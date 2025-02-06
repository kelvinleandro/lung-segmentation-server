import pytest
import numpy as np
from numpy.testing import assert_array_equal
from segmentacao.curva import inicializa_curva


def test_inicializa_curva_padrao():
    ponto = np.array([50, 50])
    curva = inicializa_curva(ponto)
    assert curva.shape == (30, 2)
    assert np.all(curva > 0)


def test_inicializa_curva_pontos_deslocados():
    ponto = np.array([100, 100])
    curva = inicializa_curva(ponto, raio=20, quantidade_pixels=50)
    assert curva.shape == (50, 2)
    assert np.all(curva > 0)


def test_inicializa_curva_pontos_negativos():
    ponto = np.array([5, 5])
    with pytest.raises(
        AssertionError, match="Os pontos da curva devem sempre ser maior que 0"
    ):
        inicializa_curva(ponto, raio=30)


def test_inicializa_curva_qtd_pontos_negativo():
    ponto = np.array([5, 5])
    with pytest.raises(
        AssertionError, match="Quantidade de pixels deve ser sempre maior que 2"
    ):
        inicializa_curva(ponto, raio=30, quantidade_pixels=-3)


def test_inicializa_curva_quantidade_minima_pixels():
    ponto = np.array([50, 50])
    curva = inicializa_curva(ponto, quantidade_pixels=2)
    assert curva.shape == (2, 2)
    assert np.all(curva > 0)


def test_inicializa_curva_raio_zero():
    ponto = np.array([30, 30])
    curva = inicializa_curva(ponto, raio=0)
    expected_output = np.tile(ponto, (30, 1))
    assert_array_equal(curva, expected_output)
