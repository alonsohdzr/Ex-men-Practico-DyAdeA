"""Recorrido del Quadtree y reconstrucción de la imagen."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from .node import QuadNode, QuadTree


def iter_leaves(root: QuadNode) -> Iterator[QuadNode]:
    """Recorrido en preorden (iterativo) que produce las hojas. Costo Theta(#nodos)."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.children is None:
            yield node
        else:
            stack.extend(reversed(node.children))


def reconstruct(tree: QuadTree) -> np.ndarray:
    """Reconstruye la imagen (H, W, C) pintando cada hoja con su color.

    Costo: Theta(#nodos) por el recorrido + Theta(H*W*C) por las escrituras.
    """
    out = np.empty((tree.height, tree.width, tree.channels), dtype=np.uint8)
    for leaf in iter_leaves(tree.root):
        out[leaf.y:leaf.y + leaf.h, leaf.x:leaf.x + leaf.w] = leaf.color
    return out


def check_partition(tree: QuadTree) -> bool:
    """Verifica que las hojas cubren cada píxel exactamente una vez (sin huecos ni solapes)."""
    cover = np.zeros((tree.height, tree.width), dtype=np.int32)
    for leaf in iter_leaves(tree.root):
        if leaf.w < 1 or leaf.h < 1:
            return False
        cover[leaf.y:leaf.y + leaf.h, leaf.x:leaf.x + leaf.w] += 1
    return bool((cover == 1).all())


def partition_overlay(tree: QuadTree, line_color: Tuple[int, int, int] = (255, 0, 0),
                      base: np.ndarray | None = None) -> np.ndarray:
    """Imagen RGB con las fronteras de las hojas dibujadas (visualización de divisiones)."""
    img = reconstruct(tree) if base is None else base
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    out = img.copy()
    for leaf in iter_leaves(tree.root):
        x2, y2 = leaf.x + leaf.w - 1, leaf.y + leaf.h - 1
        out[leaf.y, leaf.x:x2 + 1] = line_color
        out[y2, leaf.x:x2 + 1] = line_color
        out[leaf.y:y2 + 1, leaf.x] = line_color
        out[leaf.y:y2 + 1, x2] = line_color
    return out

