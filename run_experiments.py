"""Experimentos reproducibles. Genera results/results.csv.

    python run_experiments.py              # experimento completo
    python run_experiments.py --quick      # versión reducida (prueba rápida)
    python run_experiments.py --repeats 5  # más repeticiones para el tiempo (mediana)

Grupos de experimentos:
  main     tamaños 64, 128, 256 x 6 tolerancias x 4 contenidos x 2 modos de uniformidad
  scaling  tamaños 32 y 512 (para estudiar el crecimiento del costo con el tamaño)
  odd      dimensiones que NO son potencias de dos (97x61, 200x150, 255x255)
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import time

import pandas as pd
from PIL import Image

from quadtree import (bits_per_pixel, build_quadtree, compression_ratio, decode, encode,
                      encoded_size_bits, iter_leaves, max_abs_error, mse, psnr, raw_size_bytes,
                      reconstruct)
from quadtree.synthetic import CONTENT_TYPES, generate

TOLERANCES = [0, 5, 25, 100, 400, 1600]
MODES = ("naive", "integral")

GROUPS = [
    ("main", [(64, 64), (128, 128), (256, 256)], TOLERANCES, CONTENT_TYPES),
    ("scaling", [(32, 32), (512, 512)], [0, 25, 400], CONTENT_TYPES),
    ("odd", [(61, 97), (150, 200), (255, 255)], [0, 25, 400], CONTENT_TYPES),
]
QUICK_GROUPS = [("main", [(64, 64)], [0, 100], ("uniform", "blocks", "detail"))]


def png_size(img) -> int:
    buf = io.BytesIO()
    arr = img[:, :, 0] if img.shape[2] == 1 else img
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.tell()


def run_one(group, content, h, w, img, tol, mode, repeats, png_bytes):
    H, W, C = img.shape
    # Construcción: la primera corrida da las estadísticas; el tiempo es el mínimo de `repeats`
    # corridas (práctica de timeit: el mínimo es la medición menos contaminada por el sistema).
    times, tree = [], None
    for _ in range(repeats):
        t = build_quadtree(img, tol, mode)
        times.append(t.stats.build_time_s)
        tree = tree or t
    st = tree.stats

    # Recorrido (solo contar hojas) y reconstrucción (recorrido + escritura de píxeles).
    trav, rec_t = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        n_leaves = sum(1 for _ in iter_leaves(tree.root))
        trav.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        rec = reconstruct(tree)
        rec_t.append(time.perf_counter() - t0)
    assert n_leaves == st.leaves

    t0 = time.perf_counter()
    data = encode(tree)
    enc_time = time.perf_counter() - t0
    decoded_ok = bool((reconstruct(decode(data)) == rec).all())

    raw = raw_size_bytes(H, W, C)
    comp = len(data)
    err = mse(img, rec)
    return {
        "group": group,
        "image_id": f"{content}_{W}x{H}",
        "content": content,
        "width": W,
        "height": H,
        "channels": C,
        "pixels": H * W,
        "tolerance": tol,
        "uniformity_mode": mode,
        "build_time_s": min(times),
        "traverse_time_s": min(trav),
        "reconstruct_time_s": min(rec_t),
        "encode_time_s": enc_time,
        "nodes": st.nodes,
        "leaves": st.leaves,
        "internal_nodes": st.internal,
        "max_depth": st.max_depth,
        "uniformity_checks": st.checks,
        "pixel_reads": st.pixel_reads,
        "raw_bytes": raw,
        "compressed_bytes": comp,
        "compressed_bits_exact": encoded_size_bits(tree),
        "compression_ratio": compression_ratio(raw, comp),
        "space_saving_pct": 100.0 * (1.0 - comp / raw),
        "bits_per_pixel": bits_per_pixel(comp, H, W),
        "png_bytes": png_bytes,
        "ratio_vs_png": png_bytes / comp,
        "mse": err,
        "psnr_db": psnr(img, rec),
        "max_abs_error": max_abs_error(img, rec),
        "mse_within_tolerance": err <= tol,
        "lossless_when_tol0": (err == 0) if tol == 0 else "",
        "decode_roundtrip_ok": decoded_ok,
        "expands_data": comp >= raw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=os.path.join("results", "results.csv"))
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    groups = QUICK_GROUPS if a.quick else GROUPS
    total = sum(len(s) * len(t) * len(c) * len(MODES) for _, s, t, c in groups)
    k = 0
    for group, sizes, tols, contents in groups:
        for (h, w) in sizes:
            for content in contents:
                img = generate(content, h, w)
                png_bytes = png_size(img)
                for tol in tols:
                    for mode in MODES:
                        k += 1
                        # las imágenes de 512x512 son las más costosas: se miden con 2 repeticiones
                        reps = 2 if (h * w >= 512 * 512) else a.repeats
                        gc.collect()
                        gc.disable()   # evita pausas del recolector de basura durante la medición
                        try:
                            row = run_one(group, content, h, w, img, tol, mode, reps, png_bytes)
                        finally:
                            gc.enable()
                        rows.append(row)
                        print(f"[{k}/{total}] {row['image_id']:<18} tol={tol:<5} {mode:<8} "
                              f"t={row['build_time_s']:.4f}s nodos={row['nodes']:<7} "
                              f"ratio={row['compression_ratio']:.2f} mse={row['mse']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format="%.6g")
    print(f"\nResultados escritos en {a.out} ({len(df)} filas)")
    bad = df[~df["mse_within_tolerance"] | ~df["decode_roundtrip_ok"]]
    print("Verificación MSE<=tolerancia y decodificación binaria:",
          "OK en todas las filas" if bad.empty else f"FALLAS: {len(bad)}")


if __name__ == "__main__":
    main()
