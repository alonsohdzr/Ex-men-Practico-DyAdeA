"""Genera la evidencia de funcionamiento (original | reconstruida | divisiones + métricas).

    python make_evidence.py
Salida: evidence/*.png, evidence/evidence_metrics.csv, evidence/*.qt
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from quadtree import (build_quadtree, compression_ratio, encode, max_abs_error, mse,  # noqa: E402
                      partition_overlay, psnr, raw_size_bytes, reconstruct)
from quadtree.synthetic import generate  # noqa: E402

OUT = "evidence"

# (nombre, contenido, alto, ancho, tolerancia)
EXAMPLES = [
    ("ev1_blocks_256_tol0", "blocks", 256, 256, 0),
    ("ev2_detail_256_tol100", "detail", 256, 256, 100),
    ("ev3_detail_256_tol400", "detail", 256, 256, 400),
    ("ev4_noise_128_tol0", "noise", 128, 128, 0),          # caso en que el Quadtree no comprime
    ("ev5_detail_200x150_tol100", "detail", 150, 200, 100),  # dimensiones no potencia de 2
]


def evaluate(img, tol):
    tree = build_quadtree(img, tol, "integral")
    data = encode(tree)
    rec = reconstruct(tree)
    H, W, C = img.shape
    raw = raw_size_bytes(H, W, C)
    m = {
        "tolerancia": tol, "ancho": W, "alto": H, "build_ms": round(tree.stats.build_time_s * 1e3, 2),
        "nodos": tree.stats.nodes, "hojas": tree.stats.leaves, "prof_max": tree.stats.max_depth,
        "bytes_original": raw, "bytes_quadtree": len(data),
        "razon_compresion": round(compression_ratio(raw, len(data)), 3),
        "mse": round(mse(img, rec), 3), "psnr_db": round(psnr(img, rec), 2),
        "error_max": max_abs_error(img, rec),
    }
    return tree, data, rec, m


def figure(name, content, img, tol):
    tree, data, rec, m = evaluate(img, tol)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.6))
    ax[0].imshow(img)
    ax[0].set_title(f"Original ({m['ancho']}×{m['alto']}, {content})")
    ax[1].imshow(rec)
    ax[1].set_title(f"Reconstruida (τ = {tol})")
    ax[2].imshow(partition_overlay(tree))
    ax[2].set_title(f"Divisiones del Quadtree ({m['hojas']} hojas)")
    for a in ax:
        a.axis("off")
    txt = (f"nodos={m['nodos']}  hojas={m['hojas']}  prof.={m['prof_max']}  "
           f"original={m['bytes_original']} B  quadtree={m['bytes_quadtree']} B  "
           f"razón={m['razon_compresion']}  MSE={m['mse']}  PSNR={m['psnr_db']} dB  "
           f"construcción={m['build_ms']} ms")
    fig.suptitle(txt, fontsize=8, y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(OUT, name + ".qt"), "wb") as f:
        f.write(data)
    return {"evidencia": name, "contenido": content, **m}


def gallery(content="detail", size=256, tols=(0, 25, 100, 400, 1600)):
    img = generate(content, size, size)
    fig, ax = plt.subplots(2, len(tols) + 1, figsize=(2.6 * (len(tols) + 1), 5.6))
    ax[0, 0].imshow(img)
    ax[0, 0].set_title("Original", fontsize=9)
    ax[1, 0].axis("off")
    for j, tol in enumerate(tols, start=1):
        tree, _, rec, m = evaluate(img, tol)
        ax[0, j].imshow(rec)
        ax[0, j].set_title(f"τ={tol}\nMSE={m['mse']}  razón={m['razon_compresion']}", fontsize=8)
        ax[1, j].imshow(partition_overlay(tree))
        ax[1, j].set_title(f"{m['hojas']} hojas", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"gallery_{content}_{size}.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, content, h, w, tol in EXAMPLES:
        img = generate(content, h, w)
        rows.append(figure(name, content, img, tol))
        print(rows[-1])
    gallery()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "evidence_metrics.csv"), index=False)
    print(f"Evidencia escrita en {OUT}/")


if __name__ == "__main__":
    main()

