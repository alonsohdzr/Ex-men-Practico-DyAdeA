"""Estructura de datos del Quadtree (implementación propia).

Cada nodo representa una región rectangular de la imagen:

* ``(x, y)``  esquina superior izquierda (columna, fila) de la región.
* ``(w, h)``  ancho y alto de la región (siempre >= 1: nunca hay regiones vacías).
* ``color``   tupla de ``C`` enteros 0..255 con el color representativo (solo hojas).
* ``children`` ``None`` en las hojas; en nodos internos, tupla con 2 o 4 hijos en
  el orden NO, NE, SO, SE (se omiten los cuadrantes vacíos cuando ``w`` o ``h`` = 1).
"""

from __future__ import annotations

from typing import Optional, Tuple

Color = Tuple[int, ...]


class QuadNode:
    """Nodo del Quadtree. Se usa ``__slots__`` para reducir memoria por nodo."""

    __slots__ = ("x", "y", "w", "h", "color", "children")

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Optional[Color] = None,
        children: Optional[Tuple["QuadNode", ...]] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.children = children

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def area(self) -> int:
        return self.w * self.h

    def __repr__(self) -> str:  # pragma: no cover - ayuda de depuración
        kind = "Hoja" if self.is_leaf else "Interno"
        return f"QuadNode({kind}, x={self.x}, y={self.y}, w={self.w}, h={self.h}, color={self.color})"


class QuadTree:
    """Contenedor del árbol y los metadatos necesarios para reconstruir la imagen."""

    def __init__(self, root: QuadNode, height: int, width: int, channels: int,
                 tolerance: float, mode: str, stats: "BuildStats") -> None:
        self.root = root
        self.height = height
        self.width = width
        self.channels = channels
        self.tolerance = tolerance
        self.mode = mode
        self.stats = stats


class BuildStats:
    """Contadores de la construcción (se usan en el análisis experimental)."""

    __slots__ = ("nodes", "leaves", "max_depth", "checks", "pixel_reads", "build_time_s")

    def __init__(self) -> None:
        self.nodes = 0          # nodos totales (internos + hojas)
        self.leaves = 0         # nodos hoja
        self.max_depth = 0      # profundidad máxima (raíz = 0)
        self.checks = 0         # evaluaciones del criterio de uniformidad
        self.pixel_reads = 0    # lecturas de píxeles hechas por el criterio
        self.build_time_s = 0.0

    @property
    def internal(self) -> int:
        return self.nodes - self.leaves

