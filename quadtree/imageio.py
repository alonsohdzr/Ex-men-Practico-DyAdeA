"""Lectura y escritura de imágenes (Pillow solo se usa para E/S de archivos)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Carga una imagen como matriz uint8 (H, W, C): C=3 (RGB) o C=1 (escala de grises)."""
    with Image.open(path) as im:
        im = im.convert("L" if grayscale else "RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return arr[:, :, None] if arr.ndim == 2 else arr


def save_image(path: str, arr: np.ndarray) -> None:
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)

