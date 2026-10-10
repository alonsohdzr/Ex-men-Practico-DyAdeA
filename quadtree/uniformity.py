"""Criterio de uniformidad y estadísticos de región.

Definición matemática
---------------------
Sea ``R`` una región con ``n = w*h`` píxeles ``p_i in {0..255}^C``. Su color
representativo es la media por canal redondeada al entero más cercano::

    m_c = floor((2*S1_c + n) / (2*n)),      S1_c = sum_i p_{i,c}

El error cuadrático de la región respecto a ese color es::

    SSE(R) = sum_c sum_i (p_{i,c} - m_c)^2 = sum_c ( S2_c - 2 m_c S1_c + n m_c^2 )

con ``S2_c = sum_i p_{i,c}^2``. La región es **uniforme** si::

    MSE(R) = SSE(R) / (n*C)  <=  tau        (tau = tolerancia, tau >= 0)

Como la comparación se hace con el color *almacenado* (ya redondeado) y con
aritmética entera exacta, el MSE global de la imagen reconstruida es un
promedio ponderado de los MSE de las hojas y por tanto ``MSE_global <= tau``.
Con ``tau = 0`` la región debe ser exactamente constante (compresión sin pérdida).

Dos implementaciones del cálculo de ``S1`` y ``S2``:

* ``NaiveStats``    recorre los píxeles de la región: costo Theta(n*C) por consulta.
* ``IntegralStats`` usa imágenes integrales (tablas de sumas acumuladas):
  Theta(N*C) de preprocesamiento una sola vez y Theta(C) por consulta.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class RegionStats:
    """Interfaz común: ``evaluate`` devuelve (color medio entero, SSE entero)."""

    def __init__(self, img: np.ndarray) -> None:
        self.img = img
        self.H, self.W, self.C = img.shape
        self.checks = 0
        self.pixel_reads = 0

    def sums(self, x: int, y: int, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def evaluate(self, x: int, y: int, w: int, h: int) -> Tuple[np.ndarray, int]:
        n = w * h
        s1, s2 = self.sums(x, y, w, h)
        m = (2 * s1 + n) // (2 * n)                       # media redondeada (entera)
        sse = int((s2 - 2 * m * s1 + n * m * m).sum())    # SSE exacto respecto a m
        self.checks += 1
        return m, sse


class NaiveStats(RegionStats):
    """Recorre explícitamente la región: Theta(w*h*C) por llamada."""

    def sums(self, x, y, w, h):
        reg = self.img[y:y + h, x:x + w].reshape(-1, self.C).astype(np.int64)
        self.pixel_reads += w * h
        return reg.sum(axis=0), (reg * reg).sum(axis=0)


class IntegralStats(RegionStats):
    """Sumas acumuladas 2D de ``p`` y ``p^2``: Theta(C) por llamada."""

    def __init__(self, img: np.ndarray) -> None:
        super().__init__(img)
        a = img.astype(np.int64)
        self.S1 = np.zeros((self.H + 1, self.W + 1, self.C), dtype=np.int64)
        self.S2 = np.zeros((self.H + 1, self.W + 1, self.C), dtype=np.int64)
        self.S1[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
        self.S2[1:, 1:] = (a * a).cumsum(axis=0).cumsum(axis=1)
        self.pixel_reads += self.H * self.W   # preprocesamiento: una lectura por píxel

    def sums(self, x, y, w, h):
        x2, y2 = x + w, y + h
        s1 = self.S1[y2, x2] - self.S1[y, x2] - self.S1[y2, x] + self.S1[y, x]
        s2 = self.S2[y2, x2] - self.S2[y, x2] - self.S2[y2, x] + self.S2[y, x]
        return s1, s2


def make_stats(img: np.ndarray, mode: str) -> RegionStats:
    if mode == "naive":
        return NaiveStats(img)
    if mode == "integral":
        return IntegralStats(img)
    raise ValueError(f"modo de uniformidad desconocido: {mode!r} (use 'naive' o 'integral')")

