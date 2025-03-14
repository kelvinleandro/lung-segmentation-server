import logging

from segmentacao.classificacao import probabilidade_classes, calcula_ocorrencias_classes
import numpy as np
import time
import numba
from segmentacao.curva import (
    crisp_inicial,
    inicializa_curva,
    adicionar_pontos,
    remover_pontos,
)
from segmentacao.energia import energia_externa, minimiza_energia
from segmentacao.carregar import carregar_imagem

logger = logging.getLogger(__name__)


@numba.njit(parallel=True)
def minimize_curve(curva, energia_crisp, area_de_busca, w_adapt, w_cont):
    nova_curva = np.copy(curva)
    for i in numba.prange(len(nova_curva)):
        nova_curva[i] = minimiza_energia(
            curva,
            i,
            energia_crisp,
            area_de_busca=area_de_busca,
            w_adapt=w_adapt,
            w_cont=w_cont,
        )
    return nova_curva


class MCACrisp:
    def __init__(
        self,
        imagem_hu,
        y_min,
        y_max,
        x_min,
        x_max,
        quantidade_pixels=30,
        raio=30,
        w_cont=0.6,
        w_adapt=0.1,
        d_max=10.0,
        area_de_busca=9,
        alpha=20,
        early_stop=0.2,
    ):
        self.img = imagem_hu
        self.centro = crisp_inicial(self.img, y_min, y_max, x_min, x_max)
        self.curva = inicializa_curva(
            self.centro, quantidade_pixels=quantidade_pixels, raio=raio
        )
        self.ocorrencias = calcula_ocorrencias_classes(self.img)
        self.probabilidades = probabilidade_classes(self.ocorrencias)
        self.w_cont = w_cont
        self.w_adapt = w_adapt
        self.alpha = alpha
        self.energia_crisp = energia_externa(self.img, self.probabilidades)
        self.area_de_busca = area_de_busca
        self.d_max = d_max
        self.early_stop = early_stop
        self.curvas = []

    @staticmethod
    def perim(points):
        distances = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        return np.sum(distances)

    def step(self):
        self.curva = minimize_curve(
            self.curva,
            self.energia_crisp,
            self.area_de_busca,
            self.w_adapt,
            self.w_cont,
        )
        self.curva = remover_pontos(self.curva, alpha=self.alpha)
        self.curva = adicionar_pontos(self.curva, imagem=self.img, d_max=self.d_max)
        return self.curva

    def process(self, max_iterations=500):
        logger.info("Starting segmentation process")
        start_time = time.perf_counter()
        curvas = []

        for i in range(max_iterations):
            iter_start = time.perf_counter()

            curva = self.step()
            self.curvas.append(np.copy(self.curva))

            if len(self.curvas) > 1:
                current_perim = self.perim(self.curva)
                last_perim = self.perim(self.curvas[-2])

                if (
                    len(curvas) > 50
                    and np.abs(current_perim - last_perim) / last_perim
                    < self.early_stop
                ):
                    if self.area_de_busca > 1:
                        self.area_de_busca -= 2
                        self.d_max -= 1
                        logger.info(f"Adjusted area_de_busca to {self.area_de_busca}")
                    else:
                        logger.info("Convergence reached, stopping early")
                        break

            iter_time = time.perf_counter() - iter_start
            logger.info(f"Iteration {i + 1}/{max_iterations} - Time: {iter_time:.2f}s")

            yield self.curva
            curvas.append(curva)

        total_time = time.perf_counter() - start_time
        logger.info(f"Processing completed in {total_time:.2f}s")
