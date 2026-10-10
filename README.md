# Compresión de imágenes con Quadtree (Divide y Vencerás)

Prototipo en Python de un compresor de imágenes basado en un **Quadtree** construido
recursivamente (Divide y Vencerás). La estructura, la partición, el criterio de
uniformidad, la reconstrucción y el formato binario están implementados desde cero;
solo se usan bibliotecas para E/S de imágenes (Pillow), matrices (NumPy), gráficas
(Matplotlib), tablas (pandas) y para generar el PDF (ReportLab).

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `quadtree/node.py` | Nodo `QuadNode`, contenedor `QuadTree` y contadores `BuildStats` |
| `quadtree/uniformity.py` | Criterio de uniformidad (MSE ≤ τ), versión *naive* O(n) e *integral* O(1) |
| `quadtree/builder.py` | Construcción recursiva (dividir / resolver / combinar) y partición para dimensiones arbitrarias |
| `quadtree/reconstruct.py` | Recorrido, reconstrucción, verificación de la partición y visualización de divisiones |
| `quadtree/codec.py` | Formato binario (`encode` / `decode`) y tamaño exacto en bits |
| `quadtree/metrics.py` | MSE, PSNR, error máximo, razón de compresión, bits por píxel |
| `quadtree/synthetic.py` | Generador reproducible de imágenes de prueba (semilla fija) |
| `main.py` | CLI: comprimir un archivo, usar imágenes sintéticas, decodificar `.qt` |
| `run_experiments.py` | Experimentos → `results/results.csv` |
| `make_plots.py` | Gráficas → `results/figures/` |
| `make_evidence.py` | Evidencia de funcionamiento → `evidence/` |
| `make_report.py` | Reporte técnico → `report/reporte_tecnico.pdf` |
| `tests/test_quadtree.py` | Pruebas de correctitud (13 pruebas) |

## Instalación

Requiere Python ≥ 3.10.

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate        Linux/macOS:   source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
# 1) Pruebas de correctitud
python -m unittest discover -s tests -v

# 2) Comprimir una imagen propia (RGB; añadir --gray para escala de grises)
python main.py compress ruta/imagen.png --tol 100 --out output
python main.py compress ruta/imagen.png --tol 0            # sin pérdida

# 3) Imagen sintética: uniform | blocks | detail | noise
python main.py synth detail --size 256 --tol 100
python main.py synth blocks --size 200 --height 150 --tol 25   # no potencia de 2

# 4) Decodificar el archivo binario .qt generado
python main.py decompress output/detail_256x256_tol100.qt reconstruida.png

# 5) Pipeline completo y reproducible (≈ 5-10 min en un portátil)
python run_experiments.py          # --quick para una prueba rápida de 12 filas
python make_plots.py
python make_evidence.py
python make_report.py
```

`main.py` escribe `*_original.png`, `*_reconstruida.png`, `*_divisiones.png` y `*.qt`
(el archivo comprimido real) e imprime tiempo, nodos, hojas, profundidad, tamaños,
razón de compresión, MSE y PSNR.

## Decisiones de diseño (resumen)

* **Píxeles**: arreglos `uint8` de forma `(H, W, C)`, `C = 3` (RGB) o `C = 1` (grises).
* **Uniformidad**: una región de `n` píxeles es uniforme si su
  `MSE = SSE / (n·C) ≤ τ`, donde `SSE` se calcula contra el color *almacenado*
  (media por canal redondeada) con aritmética entera exacta. Consecuencia
  demostrable: **`MSE_global ≤ τ`**, y con `τ = 0` la compresión es sin pérdida.
* **Dimensiones arbitrarias**: una región `w×h` se parte en `⌈w/2⌉` / `⌊w/2⌋`
  columnas y `⌈h/2⌉` / `⌊h/2⌋` filas; los cuadrantes vacíos (cuando `w=1` o `h=1`)
  se omiten. Los cuadrantes son disjuntos y cubren la región exactamente.
* **Formato binario**: cabecera de 40 bits (ancho, alto, canales) + preorden con
  1 bit de bandera por nodo (omitido en regiones de 1 píxel) y `8·C` bits por hoja.
  La **razón de compresión** es `bytes(imagen sin comprimir) / bytes(archivo .qt)`;
  valores < 1 indican que el Quadtree *expande* la imagen.
* **Dos modos de uniformidad** (`naive` vs `integral`) para comparar el costo del
  chequeo de uniformidad: `integral` precalcula sumas acumuladas de `p` y `p²`.

## Resultados y reporte

* Resultados experimentales: `results/results.csv` (una fila por imagen × tolerancia × modo).
* Gráficas: `results/figures/`.
* Evidencia (original, reconstruida, divisiones, métricas): `evidence/`.
* Reporte técnico (PDF): `report/reporte_tecnico.pdf`.
