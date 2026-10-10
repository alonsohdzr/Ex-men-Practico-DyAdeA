"""Construcción del Quadtree mediante Divide y Vencerás.

* DIVIDIR   : evaluar uniformidad de la región; si no es uniforme, partirla en
              cuatro cuadrantes (``split_region``).
* RESOLVER  : llamada recursiva sobre cada cuadrante. Casos base: región uniforme
              (hoja con su color) o región de 1 píxel (siempre uniforme).
* COMBINAR  : el nodo padre referencia a los hijos y agrega sus contadores
              (nodos, hojas, profundidad).

Partición para dimensiones arbitrarias
--------------------------------------
Una región de ``w x h`` se divide en ``w1 = ceil(w/2)``, ``w2 = w - w1`` columnas y
``h1 = ceil(h/2)``, ``h2 = h - h1`` filas. Los cuatro rectángulos
``w1×h1, w2×h1, w1×h2, w2×h2`` son disjuntos y su unión es la región completa
(sin superposición ni omisión). Si ``w2 = 0`` (``w = 1``) o ``h2 = 0`` (``h = 1``)
los cuadrantes vacíos se omiten, por lo que nunca se crean nodos vacíos.
Como ``w1 < w`` cuando ``w >= 2`` (y análogo en ``h``), cada hijo tiene
estrictamente menos píxeles que su padre: la recursión termina.
"""

from __future__ import annotations

import math
import time
from typing import Iterator, Tuple

import numpy as np

from .node import BuildStats, QuadNode, QuadTree
from .uniformity import make_stats


def split_region(x: int, y: int, w: int, h: int) -> Iterator[Tuple[int, int, int, int]]:
    """Genera los cuadrantes no vacíos (NO, NE, SO, SE) de la región (x, y, w, h)."""
    w1, h1 = (w + 1) // 2, (h + 1) // 2
    w2, h2 = w - w1, h - h1
    for yy, hh in ((y, h1), (y + h1, h2)):
        for xx, ww in ((x, w1), (x + w1, w2)):
            if ww > 0 and hh > 0:
                yield xx, yy, ww, hh


def as_pixel_array(img: np.ndarray) -> np.ndarray:
    """Normaliza a un arreglo uint8 de forma (H, W, C) con C = 1 o 3."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 3):
        raise ValueError("la imagen debe ser (H, W) escala de grises o (H, W, 3) RGB")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("la imagen no puede estar vacía")
    if arr.dtype != np.uint8:
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError("los valores de píxel deben estar en 0..255")
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def build_quadtree(img: np.ndarray, tol: float = 0.0, mode: str = "integral") -> QuadTree:
    """Construye el Quadtree de ``img`` con tolerancia ``tol`` (MSE máximo por región).

    ``mode`` selecciona cómo se evalúa la uniformidad ('naive' o 'integral').
    """
    if not (math.isfinite(tol) and tol >= 0):
        raise ValueError("la tolerancia debe ser un número finito >= 0")
    arr = as_pixel_array(img)
    H, W, C = arr.shape

    t0 = time.perf_counter()
    stats = make_stats(arr, mode)              # (integral: preprocesamiento Theta(N*C))
    root, nodes, leaves, depth = _build(stats, 0, 0, W, H, 0, float(tol), C)
    elapsed = time.perf_counter() - t0

    st = BuildStats()
    st.nodes, st.leaves, st.max_depth = nodes, leaves, depth
    st.checks, st.pixel_reads, st.build_time_s = stats.checks, stats.pixel_reads, elapsed
    return QuadTree(root, H, W, C, float(tol), mode, st)


def _build(stats, x: int, y: int, w: int, h: int, depth: int, tol: float, C: int):
    """Devuelve (nodo, #nodos, #hojas, profundidad máxima del subárbol)."""
    n = w * h
    color, sse = stats.evaluate(x, y, w, h)

    # --- Casos base: región de 1 píxel o región uniforme (MSE <= tol) ---------------
    if n == 1 or sse <= tol * n * C:
        leaf = QuadNode(x, y, w, h, color=tuple(int(c) for c in color))
        return leaf, 1, 1, depth

    # --- DIVIDIR + RESOLVER: recursión sobre cada cuadrante no vacío ----------------
    children = []
    nodes, leaves, max_depth = 1, 0, depth
    for cx, cy, cw, ch in split_region(x, y, w, h):
        child, cn, cl, cd = _build(stats, cx, cy, cw, ch, depth + 1, tol, C)
        children.append(child)
        # --- COMBINAR: agregación de los contadores de los subárboles ---------------
        nodes += cn
        leaves += cl
        max_depth = max(max_depth, cd)
    return QuadNode(x, y, w, h, children=tuple(children)), nodes, leaves, max_depth

