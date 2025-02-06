import unittest
import numpy as np
from segmentacao.forca import forca_continuidade


class TestForcaContinuidade(unittest.TestCase):
    def test_zero_distancia(self):
        pontos = np.array([[0, 0], [0, 0], [0, 0]])
        indice = 1
        resultado = forca_continuidade(pontos, indice)
        self.assertAlmostEqual(resultado, 0.0)

    def test_pontos_linear(self):
        pontos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        indice = 2
        resultado = forca_continuidade(pontos, indice)
        self.assertTrue(resultado >= 0)

    def test_tipo_saida(self):
        np.random.seed(42)
        pontos = np.random.rand(5, 2)
        indice = 3
        resultado = forca_continuidade(pontos, indice)
        self.assertIsInstance(resultado, np.float64)

    def test_ultimo_ponto(self):
        pontos = np.array([[0, 0], [1, 1], [2, 2]])
        indice = 2
        resultado = forca_continuidade(pontos, indice)
        self.assertTrue(resultado >= 0)

    def test_triangulo(self):
        pontos = np.array([[0, 0], [4, 0], [4, 3]])
        indice = 0
        result = forca_continuidade(pontos, indice)
        # distancia de 5 é 1 maior que a média
        self.assertTrue(result == 1.0)

    def test_quadrado(self):
        pontos = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
        indice = 0
        # quadrado tem os lados iguais, então a distancia cancela
        result = forca_continuidade(pontos, indice)
        self.assertTrue(result == 0.0)


if __name__ == "__main__":
    unittest.main()
