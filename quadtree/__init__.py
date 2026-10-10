"""Compresión de imágenes con Quadtree (Divide y Vencerás) - implementación propia."""

from .builder import as_pixel_array, build_quadtree, split_region
from .codec import decode, encode, encoded_size_bits
from .metrics import (bits_per_pixel, compression_ratio, max_abs_error, mse, psnr,
                      raw_size_bytes)
from .node import BuildStats, QuadNode, QuadTree
from .reconstruct import check_partition, iter_leaves, partition_overlay, reconstruct

__all__ = [
    "build_quadtree", "split_region", "as_pixel_array", "encode", "decode",
    "encoded_size_bits", "reconstruct", "iter_leaves", "check_partition",
    "partition_overlay", "mse", "psnr", "max_abs_error", "raw_size_bytes",
    "compression_ratio", "bits_per_pixel", "QuadNode", "QuadTree", "BuildStats",
]

