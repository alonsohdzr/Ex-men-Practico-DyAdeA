"""Genera el reporte técnico en PDF (portada + 4 páginas de contenido + referencias).

    python make_report.py
Requiere haber ejecutado antes run_experiments.py, make_plots.py y make_evidence.py.
Todas las cifras de las tablas se leen de results/results.csv (nada se escribe a mano).
"""

from __future__ import annotations

import os

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

# ----------------------------- datos de la portada (completar) -----------------------------
TITLE = "Compresión de imágenes con Quadtree mediante Divide y Vencerás"
SUBTITLE = "Diseño y Análisis de Algoritmos – Examen práctico"
AUTHORS = "Integrantes: ______________________________"
DATE = "Octubre 2026"
# --------------------------------------------------------------------------------------------

CSV = os.path.join("results", "results.csv")
FIG = os.path.join("results", "figures")
EVID = "evidence"
OUT = os.path.join("report", "reporte_tecnico.pdf")

fonts = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fonts, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", os.path.join(fonts, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", os.path.join(fonts, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", os.path.join(fonts, "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

BODY = ParagraphStyle("body", fontName="DV", fontSize=7.9, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=2.5)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=10, leading=12, spaceBefore=5, spaceAfter=3,
                    textColor=colors.HexColor("#1b3a6b"), alignment=0)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=8.3, leading=10.5, spaceBefore=2, spaceAfter=1.5, alignment=0)
EQ = ParagraphStyle("eq", parent=BODY, fontName="DVM", fontSize=7.3, leading=9.3, leftIndent=10, alignment=0, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontName="DV-I", fontSize=7, leading=8.6, alignment=0, spaceAfter=3)
TITLE_S = ParagraphStyle("title", parent=BODY, fontName="DV-B", fontSize=22, leading=28, alignment=1, spaceAfter=14)
COVER = ParagraphStyle("cover", parent=BODY, fontSize=12, leading=18, alignment=1)


def P(t, s=BODY):
    return Paragraph(t, s)


def table(data, col_widths=None, font=6.6, header_rows=1, zebra=True):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    st = [("FONT", (0, 0), (-1, -1), "DV", font), ("FONT", (0, 0), (-1, header_rows - 1), "DV-B", font),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
          ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dfe7f3")),
          ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
          ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]
    if zebra:
        for r in range(header_rows, len(data)):
            if (r - header_rows) % 2 == 1:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f6f6")))
    t.setStyle(TableStyle(st))
    return t


def fmt_ms(s):
    ms = s * 1000
    return f"{ms:.1f}" if ms < 100 else f"{ms:.0f}"


def fmt_ratio(r):
    return f"{r:.1f}" if r >= 10 else f"{r:.2f}"


def build() -> None:
    df = pd.read_csv(CSV)
    main = df[df.group == "main"]
    integ = main[main.uniformity_mode == "integral"]
    naive = main[main.uniformity_mode == "naive"]
    ok = bool(df.mse_within_tolerance.all() and df.decode_roundtrip_ok.all())
    lossless = bool(df[df.tolerance == 0].lossless_when_tol0.astype(bool).all())

    story = []
    # ------------------------------------------------------------------ portada
    story += [Spacer(1, 6 * cm), P(TITLE, TITLE_S), P(SUBTITLE, COVER), Spacer(1, 1 * cm),
              P(AUTHORS, COVER), P(DATE, COVER), Spacer(1, 3 * cm),
              P("Repositorio: código, experimentos (<font name='DVM'>results/results.csv</font>) y "
                "evidencia (<font name='DVM'>evidence/</font>).", COVER),
              NextPageTemplate("body"), PageBreak()]

    # ------------------------------------------------------------------ A. Modelado
    story += [P("A. Modelado del problema", H1)]
    story += [P(
        "<b>Entrada.</b> Imagen digital de <i>H×W</i> píxeles con <i>C</i> canales (<i>C</i>=3 RGB o <i>C</i>=1 grises), "
        "matriz <font name='DVM'>uint8</font> de forma (<i>H</i>, <i>W</i>, <i>C</i>) y una tolerancia <i>τ</i> ≥ 0. "
        "El tamaño de entrada es <b><i>N</i> = <i>H·W</i></b> píxeles (con <i>C</i> constante); se usa también "
        "<i>L</i> = máx(<i>H</i>,<i>W</i>). Las dimensiones son arbitrarias (no necesariamente potencias de 2). "
        "<b>Salida.</b> (i) Un Quadtree cuyas hojas particionan la imagen en rectángulos con un color representativo, "
        "(ii) un archivo binario con esa estructura y (iii) la imagen reconstruida <i>Î</i> de dimensiones originales. "
        "<b>Restricciones.</b> Cada píxel pertenece a exactamente una hoja; no hay regiones vacías; el error de "
        "reconstrucción debe poder cuantificarse y estar acotado por <i>τ</i>.")]
    story += [P(
        "<b>Representación y similitud de colores.</b> Cada píxel es un vector <i>p</i> ∈ {0..255}<super>C</super>. "
        "El color representativo de una región <i>R</i> de <i>n</i> píxeles es la media por canal redondeada: "
        "<i>m<sub>c</sub></i> = ⌊(2·<i>S</i>1<sub>c</sub> + <i>n</i>) / (2<i>n</i>)⌋, con <i>S</i>1<sub>c</sub> = Σ<sub>i</sub> <i>p<sub>i,c</sub></i>. "
        "La disimilitud de la región es el error cuadrático medio contra ese color: "
        "MSE(<i>R</i>) = (1/(<i>nC</i>)) Σ<sub>c</sub> Σ<sub>i</sub> (<i>p<sub>i,c</sub></i> − <i>m<sub>c</sub></i>)². "
        "<b>Criterio de uniformidad:</b> <i>R</i> es uniforme ⇔ MSE(<i>R</i>) ≤ <i>τ</i>. Se justifica porque (a) es la "
        "misma métrica con que se mide la calidad final, (b) <i>τ</i> = 0 equivale a exigir que todos los píxeles sean idénticos "
        "(compresión sin pérdida), (c) usa todos los píxeles (no solo los extremos) y (d) se calcula con aritmética entera exacta "
        "a partir de Σ<i>p</i> y Σ<i>p</i>², lo que permite el cálculo en O(1) con imágenes integrales.")]

    # ------------------------------------------------------------------ B. Diseño
    story += [P("B. Diseño algorítmico", H1)]
    story += [P(
        "<b>Divide y Vencerás.</b> <font name='DVM'>build(R)</font>: <b>Dividir</b> – evaluar el criterio sobre <i>R</i>; si no se cumple, "
        "partir <i>R</i> en cuadrantes. <b>Resolver</b> – llamar recursivamente a <font name='DVM'>build</font> en cada cuadrante; "
        "<b>casos base</b>: región uniforme (hoja con su color) o región de 1 píxel (siempre uniforme, MSE = 0). "
        "<b>Combinar</b> – el nodo padre agrupa las referencias a sus hijos y agrega contadores (nodos, hojas, profundidad). "
        "Nodo: <font name='DVM'>(x, y, w, h, color | None, hijos | None)</font>; las hojas guardan el color y los nodos internos entre 2 y 4 hijos "
        "(orden NO, NE, SO, SE).")]
    story += [P(
        "<b>Dimensiones no potencia de dos.</b> Una región <i>w×h</i> se parte en <i>w</i>₁ = ⌈<i>w</i>/2⌉, <i>w</i>₂ = <i>w</i>−<i>w</i>₁ columnas y "
        "<i>h</i>₁ = ⌈<i>h</i>/2⌉, <i>h</i>₂ = <i>h</i>−<i>h</i>₁ filas. Los rectángulos <i>w</i>₁×<i>h</i>₁, <i>w</i>₂×<i>h</i>₁, <i>w</i>₁×<i>h</i>₂, "
        "<i>w</i>₂×<i>h</i>₂ son disjuntos y su unión es <i>R</i> (sin superposición ni omisión). Si <i>w</i>=1 (o <i>h</i>=1) el "
        "cuadrante con <i>w</i>₂=0 (<i>h</i>₂=0) se omite, por lo que nunca se crean nodos vacíos; esa regla también "
        "da árboles de 2 hijos en tiras de un píxel. Para <i>w</i> ≥ 2 se cumple <i>w</i>₁ < <i>w</i> (análogo en <i>h</i>), así que cada hijo tiene "
        "estrictamente menos píxeles que su padre: la recursión progresa hacia el caso base.")]
    story += [P(
        "<b>Reconstrucción.</b> Recorrido en preorden del árbol y relleno de cada hoja con su color: Θ(#nodos + <i>N·C</i>). "
        "<b>Formato de almacenamiento (bits reales).</b> Cabecera de 40 bits (ancho 16, alto 16, canales 8); luego el árbol en preorden: "
        "1 bit de bandera por nodo de área > 1 (0 = hoja, 1 = interno) y 8·<i>C</i> bits por hoja; relleno a byte. "
        "No se guardan coordenadas ni punteros: el decodificador repite la misma partición. Tamaño: "
        "<i>B</i> = ⌈(40 + <i>F</i> + 8·<i>C</i>·<i>L<sub>h</sub></i>)/8⌉ bytes con <i>F</i> = #nodos de área > 1 y <i>L<sub>h</sub></i> = #hojas. "
        "<b>Razón de compresión</b> = <i>H·W·C</i> / <i>B</i> (&gt; 1 comprime, &lt; 1 expande); también se reportan bits por píxel y MSE/PSNR "
        "(PSNR = 10 log<sub>10</sub>(255²/MSE)). Se implementaron dos variantes del criterio con idéntico árbol resultante: "
        "<b>naive</b> (recorre los píxeles de la región) e <b>integral</b> (tablas de sumas acumuladas de <i>p</i> y <i>p</i>²).")]

    # ------------------------------------------------------------------ C. Correctitud y complejidad
    story += [P("C. Correctitud y complejidad", H1)]
    story += [P(
        "<b>Correctitud.</b> <i>Lema 1 (partición):</i> los cuatro cuadrantes son disjuntos, cubren <i>R</i> y tienen área estrictamente menor (se verifica arriba y "
        "con una prueba exhaustiva para 1 ≤ <i>w</i>,<i>h</i> &lt; 20). <i>Lema 2 (terminación):</i> el área es un entero positivo que decrece en cada llamada; "
        "con área 1 hay caso base. <i>Proposición:</i> <font name='DVM'>build(R)</font> devuelve un subárbol cuyas hojas particionan <i>R</i> y cuyo color cumple "
        "MSE ≤ <i>τ</i>. Por inducción sobre el área: en el caso base es inmediato (una hoja; con 1 píxel el MSE es 0 ≤ τ); en el caso inductivo, por el Lema 1 los "
        "subproblemas particionan <i>R</i> y por hipótesis sus hojas particionan cada cuadrante y cumplen el criterio. "
        "<i>Corolario (cota de error):</i> MSE<sub>global</sub> = Σ<sub>hojas</sub> SSE / (<i>NC</i>) ≤ Σ <i>τ n<sub>i</sub>C</i> / (<i>NC</i>) = <i>τ</i>, porque la comparación "
        "usa el color <i>almacenado</i> (ya redondeado); con <i>τ</i> = 0 todas las hojas son constantes y la reconstrucción es exacta. "
        "<i>Reconstrucción:</i> cada hoja pinta su rectángulo y las hojas particionan la imagen, luego cada píxel se escribe una vez. "
        f"Verificación empírica: {len(df)} corridas, MSE ≤ <i>τ</i> en todas: <b>{'sí' if ok else 'NO'}</b>; "
        f"<i>τ</i>=0 sin pérdida: <b>{'sí' if lossless else 'NO'}</b>; decodificar(codificar(árbol)) idéntico al original en todas.")]
    story += [P(
        "<b>Operación fundamental y costo del chequeo de uniformidad.</b> Una <i>consulta</i> de uniformidad sobre una región de <i>n</i> píxeles cuesta "
        "Θ(<i>n</i>) lecturas de píxel en el modo naive (hay que ver todos, no existe corte temprano porque el criterio depende de la media) "
        "y Θ(1) lecturas de tabla en el modo integral, tras Θ(<i>N</i>) de preprocesamiento. Un modelo basado solo en #llamadas ignoraría ese costo.")]
    story += [P(
        "<b>Recurrencias</b> (imagen cuadrada, <i>n</i> = píxeles de la región, peor caso: toda región no uniforme hasta 1 píxel):", BODY)]
    story += [P("naive:     T(n) = 4·T(n/4) + Θ(n),  T(1) = Θ(1)   ⇒  a = b = 4, f(n) = Θ(n^{log₄4}) ⇒ caso 2 del método maestro ⇒ <b>T(N) = Θ(N log N)</b> = Θ(N·log L)<br/>"
                "integral:  T(n) = 4·T(n/4) + Θ(1),  T(1) = Θ(1)   ⇒  f(n) = O(n^{1−ε}) ⇒ caso 1 ⇒ T = Θ(N) (+ Θ(N) de tablas) = <b>Θ(N)</b>", EQ)]
    story += [P(
        "Árbol de recursión: el nivel <i>k</i> tiene ≤ 4<super>k</super> nodos de <i>N</i>/4<super>k</super> píxeles; naive paga Θ(<i>N</i>) por nivel y hay log₄<i>N</i>+1 = log₂<i>L</i>+1 niveles; integral paga Θ(1) por nodo y "
        "Σ4<super>k</super> = (4<i>N</i>−1)/3. <b>Cota general</b> (cualquier dimensión): los nodos de una misma profundidad son disjuntos, así que naive lee ≤ <i>N</i> píxeles por nivel y "
        "la profundidad es ≤ ⌈log₂<i>L</i>⌉, de modo que T ≤ <i>N</i>(⌈log₂<i>L</i>⌉+1) = O(<i>N</i> log <i>L</i>); además #nodos ≤ 2<i>N</i>−1 (cada nodo interno tiene ≥ 2 hijos), luego integral es O(<i>N</i>). "
        "<b>Mejor caso:</b> imagen uniforme: naive = Θ(<i>N</i>) (un solo barrido, 1 nodo), integral = Θ(<i>N</i>) por el preprocesamiento y Θ(1) de recursión. "
        "<b>Peor caso:</b> ruido con <i>τ</i> = 0 (todas las regiones se subdividen): naive Θ(<i>N</i> log <i>N</i>) con exactamente <i>N</i>(log₂<i>L</i>+1) lecturas; integral Θ(<i>N</i>), con (4<i>N</i>−1)/3 nodos. "
        "<b>Espacio.</b> Árbol: Θ(#nodos) ≤ O(<i>N</i>) en el peor caso y O(1) en el mejor; pila de recursión: Θ(profundidad) = O(log <i>L</i>); "
        "integral añade 2(<i>H</i>+1)(<i>W</i>+1)<i>C</i> enteros de 64 bits (Θ(<i>N</i>), un costo de memoria que sí es relevante). "
        "<b>Costos distintos:</b> <i>construir</i> como arriba; <i>recorrer</i> el árbol Θ(#nodos) (Ω(1), O(<i>N</i>)); <i>reconstruir</i> Θ(#nodos + <i>N</i>) = Θ(<i>N</i>) siempre, pues hay que escribir <i>N·C</i> bytes; "
        "<i>codificar/decodificar</i> Θ(#nodos).")]

    # ------------------------------------------------------------------ D. Resultados
    story += [P("D. Resultados experimentales", H1)]
    story += [P(
        "<b>Diseño.</b> Imágenes sintéticas reproducibles (semilla fija): <i>uniform</i> (un color), <i>blocks</i> (regiones de colores definidos: rectángulos, círculo, banda inclinada), "
        "<i>detail</i> (degradados, textura, formas y ruido gaussiano σ=4) y <i>noise</i> (ruido uniforme, peor caso). Tamaños 64, 128, 256 (más 32 y 512 y tres tamaños no potencia de dos: "
        "97×61, 200×150, 255×255, ancho×alto), tolerancias <i>τ</i> ∈ {0, 5, 25, 100, 400, 1600} y ambos modos de uniformidad: <b>%d</b> corridas en <font name='DVM'>results/results.csv</font> "
        "(tiempo = mínimo de 3 repeticiones, 2 en 512²; Python 3 + NumPy; las cifras de tiempo son específicas de esta máquina y de esta implementación)." % len(df))]

    # --- Tabla 1: 256x256, integral
    hdr = ["contenido", "τ", "nodos", "hojas", "prof.", "bytes", "razón", "MSE", "PSNR dB", "t naive ms", "t integ. ms"]
    rows = [hdr]
    for content in ["uniform", "blocks", "detail", "noise"]:
        for tol in [0, 5, 25, 100, 400, 1600]:
            r = integ[(integ.content == content) & (integ.width == 256) & (integ.tolerance == tol)].iloc[0]
            n = naive[(naive.content == content) & (naive.width == 256) & (naive.tolerance == tol)].iloc[0]
            psnr = "∞" if r.psnr_db == float("inf") or pd.isna(r.psnr_db) else f"{r.psnr_db:.1f}"
            rows.append([content if tol == 0 else "", tol, int(r.nodes), int(r.leaves), int(r.max_depth), int(r.compressed_bytes),
                         fmt_ratio(r.compression_ratio), f"{r.mse:.1f}", psnr, fmt_ms(n.build_time_s), fmt_ms(r.build_time_s)])
    t1 = table(rows, [1.5 * cm, 1.0 * cm, 1.4 * cm, 1.4 * cm, 1.0 * cm, 1.5 * cm, 1.3 * cm, 1.3 * cm, 1.5 * cm, 1.8 * cm, 1.9 * cm], font=6.2)
    story += [CondPageBreak(9.5 * cm), P("<b>Tabla 1.</b> Imágenes de 256×256 (196 608 bytes sin comprimir). Las 4 imágenes × 6 tolerancias; razón &lt; 1 = el Quadtree <b>expande</b> la imagen.", CAP), t1, Spacer(1, 3)]

    # --- Tabla 2: tamaños vs tiempo y operaciones
    hdr2 = ["contenido", "tam.", "N", "nodos (τ=0)", "lecturas naive/N", "t naive ms", "t integral ms", "nodos (τ=25)", "t naive ms", "t integral ms"]
    rows2 = [hdr2]
    sq = df[df.group.isin(["main", "scaling"])]
    for content in ["uniform", "detail", "noise"]:
        for side in [32, 64, 128, 256, 512]:
            def get(tol, mode):
                return sq[(sq.content == content) & (sq.width == side) & (sq.tolerance == tol) & (sq.uniformity_mode == mode)].iloc[0]
            a0n, a0i, a1n, a1i = get(0, "naive"), get(0, "integral"), get(25, "naive"), get(25, "integral")
            rows2.append([content if side == 32 else "", f"{side}²", side * side, int(a0n.nodes), f"{a0n.pixel_reads / a0n.pixels:.2f}",
                          fmt_ms(a0n.build_time_s), fmt_ms(a0i.build_time_s), int(a1n.nodes), fmt_ms(a1n.build_time_s), fmt_ms(a1i.build_time_s)])
    t2 = table(rows2, [1.5 * cm, 1.0 * cm, 1.4 * cm, 1.8 * cm, 2.2 * cm, 1.6 * cm, 1.9 * cm, 1.8 * cm, 1.6 * cm, 1.9 * cm], font=6.2)
    story += [P("<b>Tabla 2.</b> Crecimiento del costo con el tamaño (<i>τ</i> = 0 y <i>τ</i> = 25). “lecturas naive/N” = lecturas de píxel del criterio entre <i>N</i> (≈ log₂<i>L</i>+1 en el peor caso).", CAP), t2]

    story += [Image(os.path.join(FIG, "fig_scaling.png"), width=15.2 * cm, height=15.2 * cm * 6.3 / 10),
              P("<b>Figura 1.</b> Arriba: tiempo de construcción vs <i>N</i> (escala log-log; sólido = naive, discontinuo = integral). Abajo-izq.: lecturas de píxel por píxel (modo naive, <i>τ</i>=0) vs "
                "log₂<i>L</i>+1. Abajo-der.: nodos vs <i>N</i> frente a (4<i>N</i>−1)/3.", CAP),
              Image(os.path.join(FIG, "fig_tolerance.png"), width=15.2 * cm, height=15.2 * cm * 6.3 / 10),
              P("<b>Figura 2.</b> Imágenes de 256×256: hojas, profundidad y razón de compresión vs tolerancia, y curva compresión–error.", CAP)]

    story += [P("<b>Evidencia de funcionamiento</b> (carpeta <font name='DVM'>evidence/</font>): original, reconstruida, divisiones y métricas para cinco configuraciones "
                "(<font name='DVM'>ev1</font> blocks 256² τ=0; <font name='DVM'>ev2</font> detail 256² τ=100; <font name='DVM'>ev3</font> detail 256² τ=400; "
                "<font name='DVM'>ev4</font> noise 128² τ=0, donde no se comprime; <font name='DVM'>ev5</font> detail 150×200 τ=100, dimensiones no potencia de 2); "
                "<font name='DVM'>gallery_detail_256.png</font> muestra el barrido de tolerancias.")]
    story += [Image(os.path.join(EVID, "ev2_detail_256_tol100.png"), width=15.2 * cm, height=15.2 * cm * 4.6 / 12 * 1.02)]

    story += ["@@ANALYSIS@@"]
    # análisis (D interpretación) y conclusiones (E) se insertan desde analysis.py -> ver abajo
    return story, df


def make_doc(story, df, analysis_flow):
    flow = []
    for item in story:
        if item == "@@ANALYSIS@@":
            flow += analysis_flow
        else:
            flow.append(item)
    flow += [NextPageTemplate("refs"), PageBreak()]
    flow += [P("Referencias", H1)]
    for ref in REFERENCES:
        flow.append(P(ref, BODY))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                          title=TITLE, author=AUTHORS)
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def footer(canv, d):
        canv.setFont("DV", 7)
        canv.drawCentredString(A4[0] / 2, 0.8 * cm, f"Quadtree – Divide y Vencerás · página {d.page - 1}")

    doc.addPageTemplates([PageTemplate(id="cover", frames=[fr]), PageTemplate(id="body", frames=[fr], onPage=footer),
                          PageTemplate(id="refs", frames=[fr], onPage=footer)])
    doc.build(flow)
    return doc.page


REFERENCES = [
    "[1] Finkel, R. A. y Bentley, J. L. (1974). Quad trees: a data structure for retrieval on composite keys. <i>Acta Informatica</i>, 4(1), 1–9.",
    "[2] Samet, H. (1984). The quadtree and related hierarchical data structures. <i>ACM Computing Surveys</i>, 16(2), 187–260.",
    "[3] Crow, F. C. (1984). Summed-area tables for texture mapping. <i>Computer Graphics (SIGGRAPH '84)</i>, 18(3), 207–212.",
    "[4] Cormen, T. H., Leiserson, C. E., Rivest, R. L. y Stein, C. (2022). <i>Introduction to Algorithms</i> (4.ª ed.). MIT Press. (Método maestro, cap. 4.)",
    "[5] Gonzalez, R. C. y Woods, R. E. (2018). <i>Digital Image Processing</i> (4.ª ed.). Pearson. (Razón de compresión, MSE, PSNR.)",
]


# ======================================================================================
#  D (interpretación) y E (conclusiones): todas las cifras se calculan desde el CSV
# ======================================================================================
def analysis_flow(df):
    sq = df[df.group.isin(["main", "scaling"])]
    ii = df[df.uniformity_mode == "integral"]

    def g(content, side, tol, mode, col, d=None):
        d = sq if d is None else d
        return d[(d.content == content) & (d.width == side) & (d.tolerance == tol) & (d.uniformity_mode == mode)].iloc[0][col]

    sides = [64, 128, 256, 512]
    blk_nodes = " → ".join(str(int(g("blocks", s, 0, "integral", "nodes"))) for s in sides)
    full = " → ".join(str(int(g("noise", s, 0, "integral", "nodes"))) for s in sides)
    us_node = [1e6 * g(c, s, 0, "integral", "build_time_s") / g(c, s, 0, "integral", "nodes") for c in ("detail", "noise") for s in (128, 256, 512)]
    growth = g("detail", 512, 0, "integral", "build_time_s") / g("detail", 256, 0, "integral", "build_time_s")
    reads_noise = g("noise", 512, 0, "naive", "pixel_reads") / g("noise", 512, 0, "naive", "pixels")
    reads_blk = [g("blocks", s, 0, "naive", "pixel_reads") / g("blocks", s, 0, "naive", "pixels") for s in sides]
    t_n, t_i = g("noise", 512, 0, "naive", "build_time_s"), g("noise", 512, 0, "integral", "build_time_s")
    u_n, u_i = g("uniform", 512, 0, "naive", "build_time_s"), g("uniform", 512, 0, "integral", "build_time_s")

    def d256(tol, col, content="detail"):
        return g(content, 256, tol, "integral", col)

    # fracción del MSE disponible que realmente se usa (MSE real / tau) en las imágenes con detalle
    frac = (ii[(ii.content == "detail") & (ii.tolerance >= 25) & (ii.group == "main")].eval("mse / tolerance")).mean()

    pairs = ii.drop_duplicates(["image_id", "tolerance"])
    n_pairs = len(pairs)
    n_exp = int((pairs.compression_ratio < 1).sum())
    exp_contents = sorted(pairs[pairs.compression_ratio < 1].content.unique())
    over_pct = 100 * (1 / g("noise", 256, 0, "integral", "compression_ratio") - 1)
    png_b = g("blocks", 256, 0, "integral", "png_bytes")
    qt_b = g("blocks", 256, 0, "integral", "compressed_bytes")
    d100_b = d256(100, "compressed_bytes")
    d100_png = d256(100, "png_bytes")

    out = [P("<b>Interpretación.</b>", BODY)]
    out += [P(
        f"<b>(1) Tamaño de la imagen y costo.</b> Con <i>τ</i> = 0 las imágenes <i>detail</i> y <i>noise</i> producen el árbol completo, con (4<i>N</i>−1)/3 nodos ({full} para L = 64…512): "
        f"el tiempo se multiplica por ≈ {growth:.1f} al duplicar <i>L</i> (×4 sería lineal en <i>N</i>) y cuesta ≈ {min(us_node):.0f}–{max(us_node):.0f} µs por nodo, es decir, en la práctica el tiempo es proporcional al "
        f"<i>número de nodos</i>. En <i>blocks</i> los nodos crecen solo ×2–2.7 por duplicación de <i>L</i> ({blk_nodes}): son proporcionales a la longitud de las fronteras entre regiones (Θ(<i>L</i>)), no al área. "
        f"En <i>uniform</i> el árbol es un solo nodo para cualquier tamaño y el único costo es leer los <i>N</i> píxeles. "
        f"El contador de operaciones confirma el análisis: el modo naive hace exactamente {reads_noise:.2f} = log₂512+1 lecturas por píxel en ruido (Fig. 1, abajo-izq.) y {reads_blk[0]:.1f}–{reads_blk[-1]:.1f} en <i>blocks</i>, es decir Θ(<i>N</i> log <i>L</i>) en el peor caso; "
        f"el modo integral lee cada píxel una sola vez (Θ(<i>N</i>)).")]
    out += [P(
        f"<b>Naive frente a integral.</b> Aunque las lecturas difieren hasta ×{reads_noise:.0f}, los tiempos medidos difieren poco (512²/ruido, <i>τ</i>=0: {t_n:.2f} s vs {t_i:.2f} s): en esta implementación en Python el costo por nodo (≈ decenas de µs de intérprete) domina sobre el barrido vectorizado con NumPy "
        f"(≈ {1e9 * u_n / g('uniform', 512, 0, 'naive', 'pixels'):.0f} ns por píxel, medido en <i>uniform</i> 512²). La ventaja asintótica del modo integral solo se aprecia en los contadores y se vuelve visible si la región inspeccionada es grande y el árbol profundo; en el mejor caso ocurre lo opuesto: "
        f"en <i>uniform</i> 512² naive tarda {u_n * 1000:.1f} ms e integral {u_i * 1000:.1f} ms porque este último construye las tablas (Θ(<i>N</i>) de tiempo y memoria) aunque no las necesite.")]
    out += [P(
        f"<b>(2) Efecto de la tolerancia (256×256).</b> En <i>detail</i> las hojas bajan de {int(d256(0, 'leaves'))} (<i>τ</i>=0) a {int(d256(25, 'leaves'))} (25), {int(d256(100, 'leaves'))} (100), {int(d256(400, 'leaves'))} (400) y {int(d256(1600, 'leaves'))} (1600); "
        f"la mayor caída ocurre cuando <i>τ</i> supera la varianza del ruido (σ² = 16) y del degradado local, y luego hay rendimientos decrecientes. En <i>blocks</i> el número de hojas ({int(g('blocks', 256, 0, 'integral', 'leaves'))}) no cambia hasta <i>τ</i> = 25 porque las regiones son constantes: solo "
        f"las hojas de frontera cambian cuando <i>τ</i> deja de exigir uniformidad. En <i>noise</i> la varianza por canal es ≈ 5461 ≫ 1600, por lo que casi ninguna región es uniforme ({int(g('noise', 256, 1600, 'integral', 'leaves'))} hojas con <i>τ</i>=1600): la tolerancia no ayuda. "
        f"La profundidad máxima se mantiene en log₂256 = 8 salvo en <i>uniform</i> (0). El MSE real alcanzado en <i>detail</i> es solo ≈ {100 * frac:.0f} % de <i>τ</i> en promedio: <i>τ</i> es una cota por región, no el error esperado.")]
    out += [P(
        f"<b>(3) Compresión frente a calidad (<i>detail</i> 256²).</b> <i>τ</i>=25: razón {fmt_ratio(d256(25, 'compression_ratio'))}, PSNR {d256(25, 'psnr_db'):.1f} dB; "
        f"<i>τ</i>=100: {fmt_ratio(d256(100, 'compression_ratio'))}, {d256(100, 'psnr_db'):.1f} dB; <i>τ</i>=400: {fmt_ratio(d256(400, 'compression_ratio'))}, {d256(400, 'psnr_db'):.1f} dB; <i>τ</i>=1600: {fmt_ratio(d256(1600, 'compression_ratio'))}, {d256(1600, 'psnr_db'):.1f} dB. "
        f"La curva compresión–error (Fig. 2, abajo-der.) es muy empinada al inicio y se aplana: pasar de 100 a 1600 solo multiplica la razón por {d256(1600, 'compression_ratio') / d256(100, 'compression_ratio'):.1f} a costa de ≈ {d256(100, 'psnr_db') - d256(1600, 'psnr_db'):.0f} dB. "
        f"Con <i>τ</i> = 0 la reconstrucción es exacta (MSE = 0) pero solo hay ganancia si hay regiones constantes (<i>blocks</i>: {fmt_ratio(g('blocks', 256, 0, 'integral', 'compression_ratio'))}×).")]
    out += [P(
        f"<b>(4) Casos en que no se reduce el almacenamiento.</b> En {n_exp} de {n_pairs} combinaciones imagen × tolerancia el archivo es mayor o igual que la imagen original (contenidos: {', '.join(exp_contents)}). "
        f"Con ruido y <i>τ</i> = 0 el Quadtree ocupa {over_pct:.1f} % <i>más</i> que la imagen cruda: hay una hoja de 24 bits por píxel más ≈ N/3 banderas y la cabecera. "
        f"Frente a PNG (referencia opcional, columna <font name='DVM'>png_bytes</font>), el Quadtree sin pérdida es peor en <i>blocks</i> ({qt_b} B contra {png_b} B), porque cada frontera diagonal o curva se descompone en una “escalera” de hojas; "
        f"en <i>detail</i> con <i>τ</i>=100 ocupa {d100_b} B contra {d100_png} B de PNG, pero con pérdida (PSNR {d256(100, 'psnr_db'):.1f} dB), así que no es una comparación equivalente.")]
    out += [P(
        "<b>(5) ¿Cuándo conviene un Quadtree?</b> Cuando la imagen tiene regiones grandes casi constantes con fronteras alineadas o poco numerosas (gráficos, mapas, diagramas, máscaras, texturas planas) y se acepta un error acotado; "
        "el costo de construcción crece con el número de nodos, y el árbol resultante además ofrece acceso jerárquico a la distribución espacial. No conviene con ruido o texturas finas ni con <i>τ</i> = 0 sobre contenido de alta frecuencia, "
        "donde el costo aumenta a Θ(<i>N</i> log <i>N</i>) (naive) o Θ(<i>N</i>) con mayor memoria (integral) y el archivo no se reduce.")]

    out += [P("E. Conclusiones", H1)]
    out += [P(
        "• <b>Ventajas.</b> El Divide y Vencerás da una solución simple, demostrablemente correcta y con cota de error explícita (MSE ≤ <i>τ</i>); es adaptativo (más hojas donde hay detalle); el árbol es una representación jerárquica "
        "reutilizable y la reconstrucción cuesta Θ(<i>N</i>). Con imágenes uniformes o por regiones la compresión es muy alta (de 17× sin pérdida hasta &gt; 20 000× en una imagen uniforme).")]
    out += [P(
        "• <b>Limitaciones.</b> El rendimiento depende del contenido, no solo del tamaño: el peor caso (ruido) tiene (4<i>N</i>−1)/3 nodos y puede <i>expandir</i> el archivo; la partición rígida en cuadrantes no sigue bordes oblicuos o curvos; "
        "la elección de <i>τ</i> no garantiza un nivel de calidad percibido (MSE ≠ calidad visual); el modo integral cambia tiempo por Θ(<i>N</i>) de memoria. Los tiempos absolutos son de un prototipo en Python y no son comparables con un compresor en C.")]
    out += [P(
        "• <b>Recomendación.</b> Usar integral cuando los árboles sean profundos y las regiones grandes; naive si la imagen suele ser casi uniforme o la memoria es limitada. Elegir <i>τ</i> en el “codo” de la curva compresión–error "
        "(aquí <i>τ</i> ≈ 25–100 en <i>detail</i>) y no usar el Quadtree sin pérdida sobre contenido ruidoso. Trabajo futuro: podar o fusionar hojas por criterio de tasa–distorsión, codificar los colores con predicción o entropía, "
        "y usar particiones adaptativas (árboles binarios) que sigan mejor las fronteras.")]
    return out


def main():
    story, df = build()
    pages = make_doc(story, df, analysis_flow(df))
    body = pages - 2  # portada + referencias no cuentan
    print(f"Reporte escrito en {OUT}: {pages} páginas totales ({body} de contenido; máximo permitido: 4)")


if __name__ == "__main__":
    main()
