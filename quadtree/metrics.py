"""Métricas de calidad y compresión."""

from __future__ import annotations

import math

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Error cuadrático medio sobre todos los píxeles y canales."""
    d = a.astype(np.int64) - b.astype(np.int64)
    return float((d * d).mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR en dB (inf si las imágenes son idénticas)."""
    e = mse(a, b)
    return math.inf if e == 0 else 10.0 * math.log10(255.0 ** 2 / e)


def max_abs_error(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())


def raw_size_bytes(height: int, width: int, channels: int) -> int:
    """Tamaño de la imagen sin comprimir (8 bits por canal)."""
    return height * width * channels


def compression_ratio(raw_bytes: int, compressed_bytes: int) -> float:
    """Razón de compresión = tamaño original / tamaño comprimido (>1 comprime, <1 expande)."""
    return raw_bytes / compressed_bytes


def bits_per_pixel(compressed_bytes: int, height: int, width: int) -> float:
    return 8.0 * compressed_bytes / (height * width)

