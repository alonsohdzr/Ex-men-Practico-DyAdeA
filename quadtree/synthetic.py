"""Generadores reproducibles de imágenes sintéticas (semilla fija).

Las coordenadas se normalizan a [0, 1], por lo que una misma "escena" se genera
a cualquier tamaño (64, 128, 256, ... o dimensiones no potencias de dos).
"""

from __future__ import annotations

import numpy as np

CONTENT_TYPES = ("uniform", "blocks", "detail", "noise")
SEED = 20240607


def _grid(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy + 0.5) / h, (xx + 0.5) / w  # v, u en (0, 1)


def uniform(h: int, w: int) -> np.ndarray:
    """Imagen de un único color."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = (120, 180, 90)
    return img


def blocks(h: int, w: int) -> np.ndarray:
    """Regiones de colores definidos: fondo, cuadrantes irregulares, rectángulo, círculo, triángulo."""
    v, u = _grid(h, w)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = (235, 235, 225)
    img[(v < 0.37) & (u < 0.41)] = (200, 40, 40)
    img[(v < 0.37) & (u >= 0.41)] = (40, 90, 200)
    img[(v >= 0.37) & (u >= 0.62)] = (60, 170, 80)
    img[(v >= 0.55) & (v < 0.88) & (u >= 0.08) & (u < 0.45)] = (240, 200, 30)
    img[(u - 0.28) ** 2 + (v - 0.72) ** 2 < 0.012] = (120, 40, 150)
    img[(v > 0.45) & (v < 0.95) & (u > 0.62 + (v - 0.45) * 0.3) & (u < 0.62 + (v - 0.45) * 0.3 + 0.12)] = (20, 20, 20)
    return img


def detail(h: int, w: int, seed: int = SEED) -> np.ndarray:
    """Degradados suaves, textura sinusoidal, formas con bordes nítidos y ruido leve."""
    rng = np.random.default_rng(seed)
    v, u = _grid(h, w)
    r = 255 * u
    g = 255 * v
    b = 128 + 70 * np.sin(2 * np.pi * 2 * u) * np.cos(2 * np.pi * 1.5 * v)
    tex = 14 * np.sin(2 * np.pi * 18 * u) * np.sin(2 * np.pi * 14 * v)
    img = np.stack([r + tex, g + tex, b + tex], axis=2)
    disk = (u - 0.65) ** 2 + (v - 0.35) ** 2 < 0.05
    img[disk] = (255, 230, 60)
    stripes = (v > 0.7) & (np.floor(u * 12) % 2 == 0)
    img[stripes] = (30, 30, 90)
    img += rng.normal(0.0, 4.0, size=img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def noise(h: int, w: int, seed: int = SEED) -> np.ndarray:
    """Ruido uniforme independiente por píxel y canal (peor caso: sin regiones uniformes)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def generate(content: str, h: int, w: int) -> np.ndarray:
    if content == "uniform":
        return uniform(h, w)
    if content == "blocks":
        return blocks(h, w)
    if content == "detail":
        return detail(h, w)
    if content == "noise":
        return noise(h, w)
    raise ValueError(f"tipo de contenido desconocido: {content!r}; opciones: {CONTENT_TYPES}")

