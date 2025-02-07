import pytest
import numpy as np
from segmentacao.forca import forca_continuidade

def test_zero_distancia():
    pontos = np.array([[0, 0], [0, 0], [0, 0]])
    assert forca_continuidade(pontos, 1) == pytest.approx(0.0)

def test_pontos_linear():
    pontos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    assert forca_continuidade(pontos, 2) >= 0

def test_tipo_saida():
    np.random.seed(42)
    pontos = np.random.rand(5, 2)
    resultado = forca_continuidade(pontos, 3)
    assert isinstance(resultado, np.float64)

def test_ultimo_ponto():
    pontos = np.array([[0, 0], [1, 1], [2, 2]])
    assert forca_continuidade(pontos, 2) >= 0

def test_triangulo():
    pontos = np.array([[0, 0], [4, 0], [4, 3]])
    assert forca_continuidade(pontos, 0) == 1.0

def test_quadrado():
    pontos = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    assert forca_continuidade(pontos, 0) == 0.0
