"""Detección de jaulas: grosor de los bordes entre celdas + Union-Find.

Para cada par de celdas vecinas se mide cuánta "tinta" hay sobre la línea que
las separa (masa de oscuridad en una franja perpendicular a la línea). Un
borde de jaula es grueso y oscuro; una línea interna es fina y a veces gris.

No hay un umbral fijo: cada imagen tiene su propio grosor de línea, escala e
iluminación. Por eso las medidas de esa imagen se separan en dos grupos con
Otsu 1D (el corte que minimiza la varianza dentro de cada grupo). El contorno
exterior del tablero, que siempre es grueso, se agrega como referencia de
"grueso" para que el corte funcione aunque haya pocos bordes de jaula.

Finalmente, Union-Find une las celdas vecinas separadas por línea fina:
cada componente resultante es una jaula.

Regla extra de consistencia: cada jaula tiene EXACTAMENTE UNA etiqueta (en su
celda ancla). Se detecta qué celdas tienen texto arriba a la izquierda y se
usa SOLO para decidir las fronteras ambiguas (grosor cerca del umbral), así un
error al detectar una etiqueta no puede romper un borde claramente fino o grueso:
  - fronteras claramente finas: se unen siempre;
  - ambiguas, de la más fina a la más gruesa (como Kruskal): se unen salvo que
    eso junte dos componentes que ya tienen etiqueta cada una;
  - una componente sin etiqueta se une a su vecina por la frontera ambigua más fina.
"""

from __future__ import annotations

import cv2
import numpy as np

from .grid import line_masks
from .preprocessing import binarize, to_gray

Cell = tuple[int, int]


# ============================================================ medición
def _ink_mass(strip: np.ndarray) -> float:
    """Ancho equivalente de tinta en una franja (filas = muestras a lo largo de la línea).

    Para cada fila: suma de (fondo - gris)/fondo, con fondo = percentil 90 de
    esa fila (así una sombra no cuenta como tinta). Una línea negra de t px da
    ≈ t; una línea gris clara da bastante menos. Se toma la mediana de las filas
    para ignorar dígitos u otras manchas que crucen la franja.
    """
    strip = strip.astype(np.float32)
    bg = np.percentile(strip, 90, axis=1, keepdims=True) + 1.0
    darkness = np.clip((bg - strip) / bg, 0, None)
    darkness[darkness < 0.15] = 0  # ruido de sensor / textura del papel
    return float(np.median(darkness.sum(axis=1)))


def _segment(gray, xs, ys, i, j, side: str):
    """Franja que cruza el borde 'right' o 'bottom' de la celda (i, j).

    Solo se usa el tramo central-bajo del borde (45%-90%): la etiqueta está
    arriba a la izquierda de la celda y podría tocar el borde.
    """
    cw, ch = xs[j + 1] - xs[j], ys[i + 1] - ys[i]
    H, W = gray.shape

    def cut(a, b, size):  # recorte entero dentro de la imagen
        return max(0, int(a)), min(size, int(b) + 1)

    if side == "right":
        x, half = xs[j + 1], 0.18 * cw
        (r0, r1), (c0, c1) = cut(ys[i] + 0.45 * ch, ys[i] + 0.90 * ch, H), cut(x - half, x + half, W)
        return gray[r0:r1, c0:c1]
    y, half = ys[i + 1], 0.18 * ch
    (r0, r1), (c0, c1) = cut(y - half, y + half, H), cut(xs[j] + 0.45 * cw, xs[j] + 0.90 * cw, W)
    return gray[r0:r1, c0:c1].T


def border_strengths(rect: np.ndarray, xs, ys):
    """Mide todas las fronteras internas y el contorno exterior.

    Devuelve (internas, exteriores) con internas = {((i,j),(i2,j2)): masa}.
    """
    gray = to_gray(rect)
    n = len(xs) - 1
    inner = {}
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                inner[((i, j), (i, j + 1))] = _ink_mass(_segment(gray, xs, ys, i, j, "right"))
            if i + 1 < n:
                inner[((i, j), (i + 1, j))] = _ink_mass(_segment(gray, xs, ys, i, j, "bottom"))

    # Contorno exterior: se mide como el borde de celdas "virtuales" fuera del
    # tablero (columna -1, columna n, fila -1, fila n). La franja queda recortada
    # por el límite de la imagen, pero la línea del contorno está dentro.
    vx = np.concatenate([[2 * xs[0] - xs[1]], xs, [2 * xs[-1] - xs[-2]]])
    vy = np.concatenate([[2 * ys[0] - ys[1]], ys, [2 * ys[-1] - ys[-2]]])
    outer = []
    for k in range(n):
        outer.append(_ink_mass(_segment(gray, vx, vy, k + 1, 0, "right")))   # izquierda
        outer.append(_ink_mass(_segment(gray, vx, vy, k + 1, n, "right")))   # derecha
        outer.append(_ink_mass(_segment(gray, vx, vy, 0, k + 1, "bottom")))  # arriba
        outer.append(_ink_mass(_segment(gray, vx, vy, n, k + 1, "bottom")))  # abajo
    return inner, outer


# ============================================================ umbral
def otsu_1d(values: np.ndarray) -> float:
    """Corte que separa valores 1D en dos grupos minimizando la varianza intra-grupo."""
    v = np.sort(np.asarray(values, dtype=float))
    best, cut = np.inf, v.mean()
    for k in range(1, len(v)):
        a, b = v[:k], v[k:]
        cost = a.var() * len(a) + b.var() * len(b)
        if cost < best:
            best, cut = cost, (v[k - 1] + v[k]) / 2
    return cut


# ============================================================ Union-Find
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # compresión de camino
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list]:
        out: dict = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return list(out.values())


# ============================================================ etiquetas
def label_ink(rect: np.ndarray, xs, ys, c: int = 10, min_h: float = 0.05,
              dil: int = 3) -> np.ndarray:
    """Fracción de tinta "con forma de carácter" en la esquina superior
    izquierda de cada celda, donde se imprime la etiqueta de la jaula.

    Se borran las líneas de la grilla y solo se cuentan componentes conexas
    con altura de carácter (≥ 5 % de la celda): las manchas de ruido, por
    numerosas que sean, son mucho más bajas.
    """
    gray = to_gray(rect)
    S = gray.shape[0]
    bw = binarize(gray, block=(S // 20) | 1, c=c)
    horiz, vert = line_masks(rect)
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((dil, dil), np.uint8))
    text = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
    n = len(xs) - 1
    ink = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cw, ch = xs[j + 1] - xs[j], ys[i + 1] - ys[i]
            win = text[int(ys[i] + 0.04 * ch):int(ys[i] + 0.45 * ch),
                       int(xs[j] + 0.04 * cw):int(xs[j] + 0.70 * cw)]
            count, _, stats, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
            tall = stats[1:, cv2.CC_STAT_HEIGHT] >= min_h * ch
            ink[i, j] = stats[1:, cv2.CC_STAT_AREA][tall].sum() / max(1, win.size)
    return ink


def detect_labels(rect: np.ndarray, xs, ys, min_ink: float = 0.005):
    """Matriz booleana: ¿la celda (i, j) tiene etiqueta?

    Sin las líneas, una celda vacía tiene tinta ~0 y una etiqueta de un solo
    dígito ya supera el 1 %, así que basta un umbral bajo fijo. (Otsu aquí
    fallaba: separaba etiquetas cortas de largas en vez de vacío de texto.)
    """
    ink = label_ink(rect, xs, ys)
    return ink > min_ink, ink


# ============================================================ jaulas
def detect_cages(rect: np.ndarray, xs, ys, use_labels: bool = True):
    """Agrupa las celdas en jaulas.

    Devuelve (jaulas, info) con jaulas = lista de listas de celdas ordenadas
    (por su celda ancla) e info con las medidas y el umbral, para depurar.
    """
    n = len(xs) - 1
    cells = [(i, j) for i in range(n) for j in range(n)]
    inner, outer = border_strengths(rect, xs, ys)
    thr = otsu_1d(list(inner.values()) + outer)
    info = {"inner": inner, "outer": outer, "threshold": thr}

    uf = UnionFind(cells)
    values = np.array(list(inner.values()) + outer)
    thin_mean, thick_mean = values[values < thr].mean(), values[values >= thr].mean()
    band = 0.3 * (thick_mean - thin_mean) if use_labels else 0.0
    info["band"] = band
    edges = sorted(inner.items(), key=lambda kv: kv[1])
    ambiguous = [e for e in edges if thr - band <= e[1] < thr + band]

    for (a, b), mass in edges:
        if mass < thr - band:  # línea claramente fina -> misma jaula
            uf.union(a, b)

    if use_labels:
        labels, ink = detect_labels(rect, xs, ys)
        info["labels"], info["label_ink"] = labels, ink
        has_label = {}
        for c in cells:
            r = uf.find(c)
            has_label[r] = has_label.get(r, False) or bool(labels[c])

        def merge(a, b):
            ra, rb = uf.find(a), uf.find(b)
            uf.union(ra, rb)
            has_label[uf.find(rb)] = has_label[ra] or has_label[rb]

        # Kruskal sobre las ambiguas: no juntar dos componentes con etiqueta.
        for (a, b), _ in ambiguous:
            ra, rb = uf.find(a), uf.find(b)
            if ra != rb and not (has_label[ra] and has_label[rb]):
                merge(a, b)
        # Componentes sin etiqueta: unir por la ambigua más fina disponible.
        for (a, b), _ in ambiguous:
            ra, rb = uf.find(a), uf.find(b)
            if ra != rb and (not has_label[ra] or not has_label[rb]):
                merge(a, b)

    cages = sorted((sorted(g) for g in uf.groups()), key=lambda g: g[0])
    return cages, info
