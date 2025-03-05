import pytest
import numpy as np
from numpy.testing import assert_array_equal
from backend.segmentacao.curva import inicializa_curva
from backend.segmentacao.curva import no_pulmao
from backend.segmentacao.curva import na_curva
from backend.segmentacao.curva import adicionar_pontos

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


def test_no_pulmao_dentro():
    imagem = np.full((100, 100), -700)  # Simula um pulmão com valores dentro da faixa
    assert no_pulmao(np.array([50, 50]), imagem) == True

def test_no_pulmao_fora():
    imagem = np.full((100, 100), -300)  # Simula tecido fora do pulmão
    assert no_pulmao(np.array([50, 50]), imagem) == False

def test_no_pulmao_borda_externa():
    imagem = np.full((100, 100), -700)
    assert no_pulmao(np.array([-1, 50]), imagem) == False  # Fora dos limites
    assert no_pulmao(np.array([50, -1]), imagem) == False  # Fora dos limites
    assert no_pulmao(np.array([100, 50]), imagem) == False  # Fora dos limites
    assert no_pulmao(np.array([50, 100]), imagem) == False  # Fora dos limites


def test_no_pulmao_borda_valida():
    imagem = np.full((100, 100), -700)
    assert no_pulmao(np.array([0, 0]), imagem) == True  # No limite, mas dentro
    assert no_pulmao(np.array([99, 99]), imagem) == True  # No limite, mas dentro

def test_no_pulmao_valores_limite():
    imagem = np.full((100, 100), -1000)  # Valor exato do limite inferior
    assert no_pulmao(np.array([50, 50]), imagem) == True
    
    imagem = np.full((100, 100), -500)  # Valor exato do limite superior
    assert no_pulmao(np.array([50, 50]), imagem) == True

    imagem = np.full((100, 100), -1001)  # Apenas fora do limite inferior
    assert no_pulmao(np.array([50, 50]), imagem) == False
    
    imagem = np.full((100, 100), -499)  # Apenas fora do limite superior
    assert no_pulmao(np.array([50, 50]), imagem) == False

def test_ponto_dentro():
    curva = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])  # Quadrado
    ponto = np.array([5, 5])
    assert na_curva(ponto, curva) == True

def test_ponto_fora():
    curva = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])  # Quadrado
    ponto = np.array([15, 5])
    assert na_curva(ponto, curva) == False

def test_ponto_na_borda():
    curva = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])  # Quadrado
    ponto = np.array([10, 5])
    assert na_curva(ponto, curva) == False  # Supondo que a aresta do polígono é considerada "fora"

def test_ponto_no_vertice():
    curva = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])  # Quadrado
    ponto = np.array([0, 0])
    assert na_curva(ponto, curva) == False #Supondo que o vértice do polígono é considerado "fora"

def test_ponto_dentro_forma_irregular():
    curva = np.array([[0, 0], [5, 2], [2, 6], [-2, 6], [-5, 2]])  # Polígono irregular
    ponto = np.array([0, 3])
    assert na_curva(ponto, curva) == True

def test_ponto_fora_forma_irregular():
    curva = np.array([[0, 0], [5, 2], [2, 6], [-2, 6], [-5, 2]])  # Polígono irregular
    ponto = np.array([6, 3])
    assert na_curva(ponto, curva) == False

def test_adicionar_pontos_sem_modificacoes():
    curva = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
    imagem = np.full((100, 100), -700)
    d_max = 15  # Distância maior que os lados do quadrado
    nova_curva = adicionar_pontos(curva, imagem, d_max)
    assert_array_equal(nova_curva, curva)  # Nenhum ponto deve ser adicionado

def test_adicionar_pontos_simples():
    curva = np.array([[10, 10], [30, 10], [20,12]])
    imagem = np.full((100, 100), -700)
    d_max = 15  # Adiciona um ponto entre (10,10) e (30,10)
    nova_curva = adicionar_pontos(curva, imagem, d_max)
    esperado = np.array([[10, 10], [20, 10], [30, 10], [20, 12]])
    assert_array_equal(nova_curva, esperado)

def test_adicionar_pontos_com_ajuste():
    curva = np.array([[10, 10], [50, 10]])
    imagem = np.full((100, 100), -700)
    imagem[30, 10] = 0  # Simula uma região fora do pulmão
    d_max = 20
    nova_curva = adicionar_pontos(curva, imagem, d_max)
    assert len(nova_curva) > len(curva)  # Pelo menos um ponto novo foi adicionado

def test_adicionar_pontos_sem_espaco():
    curva = np.array([[10, 10], [30, 10], [20,12]])
    imagem = np.full((100, 100), -1000)  # Tudo fora do pulmão
    d_max = 15
    nova_curva = adicionar_pontos(curva, imagem, d_max)
    esperado = np.array([[10, 10], [20, 10], [30, 10], [20, 12]])
    assert_array_equal(nova_curva, esperado)  # Adiciona o ponto médio mesmo assim

if __name__ == "__main__":
    pytest.main(["backend/tests/segmentacao/test_curva.py"])

