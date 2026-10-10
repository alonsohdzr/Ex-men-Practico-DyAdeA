"""Formato binario del Quadtree (tamaño real de almacenamiento).

Formato (bit a bit, big-endian):

* Cabecera: ``ancho`` (16 bits) | ``alto`` (16 bits) | ``canales`` (8 bits) = 40 bits.
* Cuerpo, en preorden NO, NE, SO, SE. Para cada nodo:
    - si su área > 1: 1 bit de bandera (0 = hoja, 1 = nodo interno);
      las regiones de 1 píxel son hojas necesariamente y no gastan la bandera;
    - si es hoja: ``8*C`` bits con el color (un byte por canal).
* Relleno con ceros hasta completar un byte.

El decodificador reproduce la misma partición ``split_region`` a partir de las
dimensiones de la cabecera, por eso no hace falta guardar coordenadas, tamaños
ni punteros: la estructura se infiere de la bandera 0/1.
"""

from __future__ import annotations

from .builder import split_region
from .node import QuadNode, QuadTree

HEADER_BITS = 16 + 16 + 8


class BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, value: int, bits: int) -> None:
        self.acc = (self.acc << bits) | value
        self.n += bits
        while self.n >= 8:
            self.n -= 8
            self.buf.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def finish(self) -> bytes:
        if self.n:
            self.buf.append((self.acc << (8 - self.n)) & 0xFF)
            self.acc, self.n = 0, 0
        return bytes(self.buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, bits: int) -> int:
        v = 0
        while bits > 0:
            idx = self.pos >> 3
            if idx >= len(self.data):
                raise ValueError("flujo de bits truncado")
            off = self.pos & 7
            take = min(8 - off, bits)
            v = (v << take) | ((self.data[idx] >> (8 - off - take)) & ((1 << take) - 1))
            self.pos += take
            bits -= take
        return v


def encoded_size_bits(tree: QuadTree) -> int:
    """Tamaño exacto en bits (sin relleno) calculado a partir de los contadores."""
    flags = 0
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if node.w * node.h > 1:
            flags += 1
        if node.children:
            stack.extend(node.children)
    return HEADER_BITS + flags + tree.stats.leaves * 8 * tree.channels


def encode(tree: QuadTree) -> bytes:
    if tree.width > 0xFFFF or tree.height > 0xFFFF:
        raise ValueError("el formato admite dimensiones de hasta 65535 píxeles")
    bw = BitWriter()
    bw.write(tree.width, 16)
    bw.write(tree.height, 16)
    bw.write(tree.channels, 8)
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if node.w * node.h > 1:
            bw.write(0 if node.children is None else 1, 1)
        if node.children is None:
            for c in node.color:
                bw.write(c, 8)
        else:
            stack.extend(reversed(node.children))
    return bw.finish()


def decode(data: bytes) -> QuadTree:
    """Reconstruye el árbol desde el flujo binario (para verificar el formato)."""
    from .node import BuildStats

    br = BitReader(data)
    W, H, C = br.read(16), br.read(16), br.read(8)
    if W < 1 or H < 1 or C not in (1, 3):
        raise ValueError("cabecera inválida")
    st = BuildStats()

    def rec(x, y, w, h, depth):
        st.nodes += 1
        st.max_depth = max(st.max_depth, depth)
        internal = br.read(1) if w * h > 1 else 0
        if not internal:
            st.leaves += 1
            return QuadNode(x, y, w, h, color=tuple(br.read(8) for _ in range(C)))
        kids = tuple(rec(cx, cy, cw, ch, depth + 1) for cx, cy, cw, ch in split_region(x, y, w, h))
        return QuadNode(x, y, w, h, children=kids)

    root = rec(0, 0, W, H, 0)
    return QuadTree(root, H, W, C, tolerance=float("nan"), mode="decoded", stats=st)

