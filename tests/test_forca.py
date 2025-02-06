import unittest
import numpy as np
from segmentacao.forca import forca_continuidade
class TestForcaContinuidade(unittest.TestCase):
    def test_zero_distance(self):
        pontos = np.array([[0, 0], [0, 0], [0, 0]])
        indice = 1
        result = forca_continuidade(pontos, indice)
        self.assertAlmostEqual(result, 0.0)

    def test_linear_points(self):
        pontos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        indice = 2
        result = forca_continuidade(pontos, indice)
        self.assertTrue(result >= 0)

    def test_random_points(self):
        np.random.seed(42)
        pontos = np.random.rand(5, 2)
        indice = 3
        result = forca_continuidade(pontos, indice)
        self.assertIsInstance(result, np.float64)

    def test_last_point(self):
        pontos = np.array([[0, 0], [1, 1], [2, 2]])
        indice = 2
        result = forca_continuidade(pontos, indice)
        self.assertTrue(result >= 0)
        
    def test_triangulo(self):
        pontos = np.array([[0, 0], [4,0],[4, 3]])
        indice = 0
        result = forca_continuidade(pontos, indice)
        self.assertTrue(result == 1.0)
        
    def test_quadrado(self):
        pontos = np.array([[0, 0], [4,0],[4, 4], [0, 4]])
        indice = 0
        result = forca_continuidade(pontos, indice)
        self.assertTrue(result == 0.0)

if __name__ == '__main__':
    unittest.main()
