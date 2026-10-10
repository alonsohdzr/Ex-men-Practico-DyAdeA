"""CLI: comprime una imagen con Quadtree, guarda la reconstrucción y muestra métricas.

Uso:
    python main.py compress ruta/imagen.png --tol 100 [--gray] [--mode integral|naive] [--out salida]
    python main.py synth blocks --size 128 --tol 100          # imagen sintética
    python main.py decompress salida/imagen.qt salida/rec.png  # lee el formato binario
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from quadtree import (bits_per_pixel, build_quadtree, compression_ratio, decode, encode,
                      max_abs_error, mse, partition_overlay, psnr, raw_size_bytes, reconstruct)
from quadtree.imageio import load_image, save_image
from quadtree.synthetic import CONTENT_TYPES, generate


def run(img: np.ndarray, name: str, tol: float, mode: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    H, W, C = img.shape
    tree = build_quadtree(img, tol, mode)
    data = encode(tree)
    rec = reconstruct(tree)
    st = tree.stats
    raw = raw_size_bytes(H, W, C)
    base = os.path.join(out_dir, f"{name}_tol{tol:g}")
    save_image(base + "_original.png", img)
    save_image(base + "_reconstruida.png", rec)
    save_image(base + "_divisiones.png", partition_overlay(tree))
    with open(base + ".qt", "wb") as f:
        f.write(data)
    print(f"Imagen: {name}  {W}x{H}  canales={C}  píxeles={H * W}")
    print(f"Tolerancia (MSE max por región) = {tol:g}   modo de uniformidad = {mode}")
    print(f"Tiempo de construcción = {st.build_time_s * 1000:.2f} ms")
    print(f"Nodos = {st.nodes}  hojas = {st.leaves}  profundidad máx = {st.max_depth}")
    print(f"Tamaño original = {raw} B   Quadtree = {len(data)} B   "
          f"razón de compresión = {compression_ratio(raw, len(data)):.3f}   "
          f"bpp = {bits_per_pixel(len(data), H, W):.3f}")
    print(f"MSE = {mse(img, rec):.3f}   PSNR = {psnr(img, rec):.2f} dB   error máx = {max_abs_error(img, rec)}")
    print(f"Archivos escritos con prefijo: {base}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="comprimir un archivo de imagen")
    c.add_argument("path")
    c.add_argument("--tol", type=float, default=100.0, help="MSE máximo por región (0 = sin pérdida)")
    c.add_argument("--gray", action="store_true", help="convertir a escala de grises")
    c.add_argument("--mode", choices=("integral", "naive"), default="integral")
    c.add_argument("--out", default="output")

    s = sub.add_parser("synth", help="usar una imagen sintética reproducible")
    s.add_argument("content", choices=CONTENT_TYPES)
    s.add_argument("--size", type=int, default=128)
    s.add_argument("--height", type=int, default=None)
    s.add_argument("--tol", type=float, default=100.0)
    s.add_argument("--mode", choices=("integral", "naive"), default="integral")
    s.add_argument("--out", default="output")

    d = sub.add_parser("decompress", help="decodificar un archivo .qt a PNG")
    d.add_argument("qt")
    d.add_argument("png")

    a = p.parse_args(argv)
    if a.cmd == "compress":
        name = os.path.splitext(os.path.basename(a.path))[0]
        run(load_image(a.path, a.gray), name, a.tol, a.mode, a.out)
    elif a.cmd == "synth":
        h = a.height or a.size
        run(generate(a.content, h, a.size), f"{a.content}_{a.size}x{h}", a.tol, a.mode, a.out)
    else:
        with open(a.qt, "rb") as f:
            tree = decode(f.read())
        save_image(a.png, reconstruct(tree))
        print(f"Imagen {tree.width}x{tree.height} guardada en {a.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

