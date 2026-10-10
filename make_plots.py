"""Genera las gráficas del análisis experimental a partir de results/results.csv.

    python make_plots.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = os.path.join("results", "results.csv")
OUT = os.path.join("results", "figures")
COLORS = {"uniform": "tab:gray", "blocks": "tab:blue", "detail": "tab:green", "noise": "tab:red"}


def scaling_figure(df: pd.DataFrame) -> None:
    sq = df[df["group"].isin(["main", "scaling"])]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.3))

    for a, tol in ((ax[0, 0], 0), (ax[0, 1], 25)):
        for content, color in COLORS.items():
            for mode, ls in (("naive", "-"), ("integral", "--")):
                d = sq[(sq.content == content) & (sq.tolerance == tol) & (sq.uniformity_mode == mode)].sort_values("pixels")
                a.plot(d["pixels"], d["build_time_s"], ls, marker="o", ms=3, color=color,
                       label=f"{content} ({mode})")
        n = np.array([1024, 262144])
        t0 = sq[(sq.content == "noise") & (sq.tolerance == 0) & (sq.uniformity_mode == "integral") & (sq.pixels == 4096)]["build_time_s"].iloc[0]
        a.plot(n, t0 * n / 4096, ":", color="k", lw=1, label="referencia ∝ N")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("píxeles N = ancho·alto")
        a.set_ylabel("tiempo de construcción (s)")
        a.set_title(f"Tiempo vs tamaño, τ = {tol}")
        a.grid(alpha=0.3, which="both")
    ax[0, 0].legend(fontsize=6, ncol=2)

    a = ax[1, 0]
    for content, color in COLORS.items():
        d = sq[(sq.content == content) & (sq.tolerance == 0) & (sq.uniformity_mode == "naive")].sort_values("pixels")
        a.plot(np.sqrt(d["pixels"]), d["pixel_reads"] / d["pixels"], marker="o", color=color, label=content)
    sides = np.array([32, 64, 128, 256, 512])
    a.plot(sides, np.log2(sides) + 1, ":", color="k", label="log₂(L)+1")
    a.set_xscale("log", base=2)
    a.set_xlabel("lado L de la imagen")
    a.set_ylabel("lecturas de píxel / N (modo naive)")
    a.set_title("Operación fundamental: lecturas por píxel, τ = 0")
    a.legend(fontsize=7)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    for content, color in COLORS.items():
        d = sq[(sq.content == content) & (sq.tolerance == 0) & (sq.uniformity_mode == "integral")].sort_values("pixels")
        a.plot(d["pixels"], d["nodes"], marker="o", color=color, label=content)
    n = np.array([1024, 262144])
    a.plot(n, (4 * n - 1) / 3, ":", color="k", label="(4N−1)/3")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("píxeles N")
    a.set_ylabel("nodos totales")
    a.set_title("Nodos vs tamaño, τ = 0")
    a.legend(fontsize=7)
    a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scaling.png"), dpi=150)
    plt.close(fig)


def tolerance_figure(df: pd.DataFrame, size: int = 256) -> None:
    d0 = df[(df.group == "main") & (df.width == size) & (df.uniformity_mode == "integral")]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.3))
    for content, color in COLORS.items():
        d = d0[d0.content == content].sort_values("tolerance")
        ax[0, 0].plot(d.tolerance, d.leaves, marker="o", color=color, label=content)
        ax[0, 1].plot(d.tolerance, d.max_depth, marker="o", color=color, label=content)
        ax[1, 0].plot(d.tolerance, d.compression_ratio, marker="o", color=color, label=content)
        ax[1, 1].plot(d.mse, d.compression_ratio, marker="o", color=color, label=content)
    for a in ax.ravel():
        a.set_xscale("symlog", linthresh=1)
        a.grid(alpha=0.3, which="both")
    ax[0, 0].set_yscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 0].axhline(1.0, color="k", ls=":", lw=1)
    ax[1, 1].axhline(1.0, color="k", ls=":", lw=1)
    ax[0, 0].set(xlabel="tolerancia τ (MSE máx. por región)", ylabel="hojas", title=f"Hojas vs tolerancia ({size}×{size})")
    ax[0, 1].set(xlabel="tolerancia τ", ylabel="profundidad máxima", title="Profundidad vs tolerancia")
    ax[1, 0].set(xlabel="tolerancia τ", ylabel="razón de compresión", title="Razón de compresión vs τ (línea: 1 = sin ganancia)")
    ax[1, 1].set(xlabel="MSE de reconstrucción", ylabel="razón de compresión", title="Compresión vs error (rate–distortion)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tolerance.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV)
    scaling_figure(df)
    tolerance_figure(df)
    print(f"Figuras escritas en {OUT}")


if __name__ == "__main__":
    main()

