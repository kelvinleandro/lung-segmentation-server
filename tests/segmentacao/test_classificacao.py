import pytest
import numpy as np
from backend.segmentacao.classificacao import (
    calcula_ocorrencias_classes,
    probabilidade_classes,
)


@pytest.fixture
def amostra_imagem():
    return np.array(
        [[50, 700, -600], [-200, 1500, -900], [0, -750, 300]], dtype=np.int16
    )


@pytest.fixture
def ocorrencias_amostra():
    return np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Classe 0
            [[0, 0, 1], [0, 0, 0], [0, 1, 0]],  # Classe 1
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],  # Classe 2
            [[1, 0, 0], [0, 0, 0], [1, 0, 1]],  # Classe 3
            [[0, 1, 0], [0, 1, 0], [0, 0, 0]],  # Classe 4
        ],
        dtype=np.uint8,
    )


def teste_calcula_ocorrencias_classes(amostra_imagem):
    ocorrencias = calcula_ocorrencias_classes(amostra_imagem)
    assert ocorrencias.shape == (5, 3, 3), "A forma da saída está incorreta."
    assert ocorrencias.dtype == np.uint8, "O tipo de dados da saída está incorreto."
    assert np.all(ocorrencias >= 0), "Ocorrências não podem ser negativas."


def teste_probabilidade_classes(ocorrencias_amostra):
    probabilidades = probabilidade_classes(ocorrencias_amostra)
    assert probabilidades.shape == (5, 3, 3), "A forma da saída está incorreta."
    assert probabilidades.dtype == np.float32, (
        "O tipo de dados da saída está incorreto."
    )

    soma_probabilidades = probabilidades.sum(axis=0)
    np.testing.assert_allclose(
        soma_probabilidades,
        1.0,
        atol=1e-6,
        err_msg="A soma das probabilidades por pixel deve ser 1.",
    )

    zero_mask = ocorrencias_amostra.sum(axis=0) == 0
    assert np.all(probabilidades[:, zero_mask] == 0.2), (
        "Para pixels sem ocorrências, a probabilidade deve ser 0.2 (1/5)."
    )

if __name__ == "__main__":
    pytest.main(["backend/tests/segmentacao/test_classificacao.py"])