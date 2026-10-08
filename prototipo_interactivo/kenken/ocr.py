"""OCR de las etiquetas de las jaulas: recorte -> caracteres -> CNN -> lecturas.

Flujo para cada jaula:
  1. crop_label():     recorta la mitad superior de su celda ancla en el
                       tablero rectificado, la reescala, borra las líneas de
                       la grilla y binariza con Otsu dentro del recorte.
  2. segment_glyphs(): separa los caracteres con componentes conexas. Las
                       piezas que se superponen en x se agrupan (los puntos
                       del '÷' van con su barra) y las cajas demasiado anchas
                       (dos caracteres pegados) se cortan. Orden: izquierda a derecha.
  3. normalize_glyph(): cada carácter -> imagen 32x32 de "oscuridad" en [0,1]
                       (tinta = 1, papel = 0), centrado y manteniendo la
                       proporción. Es la entrada de la CNN.
  4. La CNN (cnn.py) da una distribución de probabilidad sobre 14 clases por carácter.
  5. decode_readings(): combina las probabilidades con la gramática de una
                       etiqueta ("dígitos + operación", y qué operaciones admite
                       el tamaño de la jaula) y devuelve las k lecturas más
                       probables con su log-probabilidad. Esas alternativas son
                       las que usa la inferencia conjunta del modelo CP.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .grid import line_masks
from .preprocessing import to_gray

# Las 14 clases de la CNN. 'x' y '×' son la misma clase (multiplicación); '/' y '÷' también.
CLASSES = list("0123456789") + ["+", "-", "*", "/"]
CLASS_INDEX = {c: k for k, c in enumerate(CLASSES)}
DIGITS = list(range(10))
GLYPH_TO_CLASS = {**{d: d for d in "0123456789"}, "+": "+", "-": "-", "−": "-", "–": "-",
                  "*": "*", "x": "*", "X": "*", "×": "*", "/": "/", "÷": "/", ":": "/"}
GLYPH_SIZE = 32
LABEL_H = 160  # alto (px) al que se reescala cada celda antes de segmentar


# ============================================================ 1. recorte
@dataclass
class LabelCrop:
    gray: np.ndarray      # zona de la etiqueta en gris, reescalada
    ink: np.ndarray       # "oscuridad" normalizada en [0,1] (1 = tinta), sin líneas
    binary: np.ndarray    # tinta binaria (Otsu local), sin líneas
    cell_h: float         # alto de la celda en el recorte reescalado (= LABEL_H)


def grid_line_mask(rect: np.ndarray) -> np.ndarray:
    """Máscara (dilatada) de las líneas de la grilla, para borrarlas del texto."""
    horiz, vert = line_masks(rect)
    return cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((5, 5), np.uint8))


def crop_label(rect: np.ndarray, lines: np.ndarray, xs, ys, cell) -> LabelCrop:
    """Zona de la etiqueta (mitad superior de la celda ancla), reescalada a una
    altura de celda fija y binarizada con Otsu SOLO dentro del recorte: así el
    umbral se adapta a la luz local y los trazos no se engordan."""
    i, j = cell
    cw, ch = xs[j + 1] - xs[j], ys[i + 1] - ys[i]
    y0, y1 = int(ys[i] + 0.03 * ch), int(ys[i] + 0.50 * ch)
    x0, x1 = int(xs[j] + 0.03 * cw), int(xs[j + 1] - 0.03 * cw)
    gray = to_gray(rect)[y0:y1, x0:x1]
    line = lines[y0:y1, x0:x1]
    f = LABEL_H / ch
    gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    line = cv2.resize(line, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return label_from_gray(gray, line)


def label_from_gray(gray: np.ndarray, line: np.ndarray | None = None) -> LabelCrop:
    """Recorte en gris (ya a escala LABEL_H) -> tinta normalizada y binaria.

    Se usa igual en el pipeline y al generar datos de entrenamiento, para que
    la CNN vea exactamente el mismo tipo de entrada en ambos casos.
    """
    if line is None:
        line = np.zeros_like(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    paper, dark = np.percentile(gray, 90), np.percentile(gray, 1)
    ink = np.clip((paper - gray.astype(np.float32)) / max(paper - dark, 20.0), 0, 1)
    ink[line > 0] = 0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary[line > 0] = 0
    if paper - dark < 25:  # recorte sin contraste: no hay texto
        binary[:] = 0
    return LabelCrop(gray, ink, binary, float(LABEL_H))


# ============================================================ 2. segmentación
def _tight(binary, box):
    """Ajusta la caja a la tinta que contiene (None si está vacía)."""
    x0, y0, x1, y1 = box
    sub = binary[y0:y1, x0:x1] > 0
    if not sub.any():
        return None
    cx, cy = np.flatnonzero(sub.any(axis=0)), np.flatnonzero(sub.any(axis=1))
    return [x0 + cx[0], y0 + cy[0], x0 + cx[-1] + 1, y0 + cy[-1] + 1]


def _split_wide(binary: np.ndarray, box, ref_h: float) -> list:
    """Divide una caja demasiado ancha para ser un carácter (glifos pegados)
    en la columna con menos tinta de su zona central, recursivamente."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    if w <= 1.1 * ref_h or w < 8:
        return [box]
    proj = (binary[y0:y1, x0:x1] > 0).sum(axis=0)
    a, b = int(0.25 * w), int(0.75 * w)
    cut = x0 + a + int(np.argmin(proj[a:b]))
    out = []
    for part in ([x0, y0, cut, y1], [cut, y0, x1, y1]):
        part = _tight(binary, part)
        if part is not None:
            out += _split_wide(binary, part, ref_h)
    return out


def segment_glyphs(binary: np.ndarray, cell_h: float) -> list[tuple[int, int, int, int]]:
    """Cajas (x0, y0, x1, y1) de cada carácter, de izquierda a derecha."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = [[s[0], s[1], s[0] + s[2], s[1] + s[3]] for s in stats[1:]
             if s[cv2.CC_STAT_AREA] >= max(4, (0.012 * cell_h) ** 2)]
    if not boxes:
        return []

    # Agrupar piezas que se superponen horizontalmente (puntos del ÷, trozos de
    # un carácter roto). Solo si una de las dos es pequeña: dos caracteres
    # completos con kerning (p. ej. '3/') también se solapan y no deben unirse.
    boxes.sort(key=lambda b: b[0])
    max_h = max(b[3] - b[1] for b in boxes)
    groups = [boxes[0]]
    for b in boxes[1:]:
        g = groups[-1]
        overlap = min(g[2], b[2]) - max(g[0], b[0])
        small = min(g[3] - g[1], b[3] - b[1]) < 0.45 * max_h
        if overlap > 0.5 * min(g[2] - g[0], b[2] - b[0]) and small:
            groups[-1] = [min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3])]
        else:
            groups.append(b)

    # Descartar lo que no tenga tamaño de carácter (ruido, restos de línea).
    groups = [g for g in groups if (g[3] - g[1]) >= 0.05 * cell_h or (g[2] - g[0]) >= 0.06 * cell_h]
    if not groups:
        return []

    # La etiqueta es una sola línea de texto: se quitan grupos lejos de la
    # línea del carácter más alto (p. ej. un número escrito a mano más abajo).
    tallest = max(groups, key=lambda g: g[3] - g[1])
    top, bottom = tallest[1], tallest[3]
    h = bottom - top
    groups = [g for g in groups if g[1] < bottom + 0.2 * h and g[3] > top - 0.2 * h]

    # Separar el texto de lo que esté muy a la derecha (otro objeto en la celda).
    out = [groups[0]]
    for g in groups[1:]:
        if g[0] - out[-1][2] > 1.2 * h:
            break
        out.append(g)

    # Glifos pegados: cortar las cajas demasiado anchas.
    final = []
    for g in out:
        final += _split_wide(binary, g, h)
    return [tuple(map(int, g)) for g in final]


# ============================================================ 3. normalización
def normalize_glyph(ink: np.ndarray, box, size: int = GLYPH_SIZE, inner: int = 24) -> np.ndarray:
    """Recorta el carácter, lo escala para que su lado mayor mida `inner` px
    (manteniendo la proporción: un '-' sigue siendo ancho y bajo) y lo centra
    en una imagen size x size. Devuelve float32 en [0, 1], tinta = 1."""
    x0, y0, x1, y1 = box
    g = ink[y0:y1, x0:x1].astype(np.float32)
    h, w = g.shape
    scale = inner / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = g
    return out


SPLIT_PENALTY = 3.0  # costo (en log-prob) de usar una segmentación alternativa


def segmentation_hypotheses(binary: np.ndarray, boxes: list, cell_h: float):
    """Segmentación principal + alternativas con UNA corrección cada una:
    partir un trozo en dos (dos caracteres pegados, p. ej. '1-') o unir dos
    trozos vecinos (un carácter partido). Devuelve [(cajas, penalización)].

    La CNN y la gramática deciden después cuál encaja mejor; la penalización
    hace que una alternativa solo gane si es claramente más verosímil.
    """
    hyps = [(list(boxes), 0.0)]
    if not boxes:
        return hyps
    ref_h = max(b[3] - b[1] for b in boxes)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        w = x1 - x0
        if w >= 0.45 * ref_h:
            proj = (binary[y0:y1, x0:x1] > 0).sum(axis=0)
            a, b = int(0.3 * w), int(0.8 * w)
            if b > a:
                cut = x0 + a + int(np.argmin(proj[a:b]))
                parts = [_tight(binary, [x0, y0, cut, y1]), _tight(binary, [cut, y0, x1, y1])]
                if all(p is not None for p in parts):
                    hyps.append((boxes[:k] + [tuple(map(int, p)) for p in parts] + boxes[k + 1:],
                                 SPLIT_PENALTY))
    for k in range(len(boxes) - 1):
        a, b = boxes[k], boxes[k + 1]
        merged = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        hyps.append((boxes[:k] + [merged] + boxes[k + 2:], SPLIT_PENALTY))
    return hyps


def extract_label_glyphs(rect: np.ndarray, xs, ys, cages, lines: np.ndarray | None = None,
                         alternatives: bool = True):
    """Para cada jaula: lista de hipótesis [(glifos (m, 32, 32), penalización)].

    La primera hipótesis es siempre la segmentación principal (penalización 0).
    """
    if lines is None:
        lines = grid_line_mask(rect)
    out = []
    for cage in cages:
        lc = crop_label(rect, lines, xs, ys, min(map(tuple, cage)))
        boxes = segment_glyphs(lc.binary, lc.cell_h)
        hyps = segmentation_hypotheses(lc.binary, boxes, lc.cell_h) if alternatives else [(boxes, 0.0)]
        out.append([(np.stack([normalize_glyph(lc.ink, b) for b in bxs]) if bxs
                     else np.zeros((0, GLYPH_SIZE, GLYPH_SIZE), np.float32), pen)
                    for bxs, pen in hyps])
    return out


# ============================================================ 5. decodificación
def allowed_ops(cage_size: int) -> list[str]:
    """Operaciones posibles según el tamaño de la jaula (reglas de KenKen)."""
    if cage_size == 1:
        return ["="]
    if cage_size == 2:
        return ["+", "-", "*", "/"]
    return ["+", "*"]


def decode_readings(log_probs: np.ndarray, cage_size: int, n: int, k: int = 5) -> list[dict]:
    """Las k lecturas (objetivo, operación) más probables de una etiqueta.

    log_probs: (num_glifos, 14) log-probabilidades de la CNN.
    Gramática: jaula de 1 celda -> solo dígitos; si no -> dígitos + 1 operación
    final, restringida por allowed_ops(). Sin ceros a la izquierda. La
    probabilidad de una lectura es el producto de las de sus caracteres
    (suma de log-probs), suponiendo independencia entre caracteres.
    Se descartan lecturas imposibles para n (p. ej. '÷' con objetivo > n).
    """
    from .instance import Cage, _cage_errors  # import local: evita ciclo

    ops = allowed_ops(cage_size)
    n_digits = len(log_probs) if ops == ["="] else len(log_probs) - 1
    if n_digits < 1:
        return []

    # Opciones por posición, ordenadas de más a menos probable: (log-prob, símbolo).
    slots = [sorted(((log_probs[p, d], str(d)) for d in DIGITS if not (p == 0 and d == 0)),
                    reverse=True) for p in range(n_digits)]
    if ops != ["="]:
        slots.append(sorted(((log_probs[-1, CLASS_INDEX[o]], o) for o in ops), reverse=True))

    def score(idx):
        return sum(slots[p][q][0] for p, q in enumerate(idx))

    # k-best sobre el producto cartesiano: se parte de la mejor combinación y
    # se expanden "vecinos" (una posición pasa a su siguiente opción).
    cells = [(0, c) for c in range(cage_size)]  # celdas en fila, solo para validar rangos
    start = tuple(0 for _ in slots)
    heap, seen, results = [(-score(start), start)], {start}, []
    while heap and len(results) < k and len(seen) < 5000:
        neg, idx = heapq.heappop(heap)
        symbols = [slots[p][q][1] for p, q in enumerate(idx)]
        op = "=" if ops == ["="] else symbols[-1]
        target = int("".join(symbols if ops == ["="] else symbols[:-1]))
        if not _cage_errors(Cage(cells, target, op), n):
            results.append({"target": target, "op": op, "logp": float(-neg)})
        for p in range(len(slots)):
            if idx[p] + 1 < len(slots[p]):
                nxt = idx[:p] + (idx[p] + 1,) + idx[p + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (-score(nxt), nxt))
    return results


def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def reading_label(r: dict) -> str:
    return str(r["target"]) if r["op"] == "=" else f'{r["target"]}{r["op"]}'


def prob(r: dict) -> float:
    return math.exp(r["logp"])
