"""Detección del tamaño n del tablero y de la posición de las líneas de la grilla.

Trabaja sobre el tablero ya rectificado (cuadrado). Idea:
  1. Binarizar y quedarse solo con trazos largos horizontales / verticales
     (apertura morfológica con un kernel alargado): las etiquetas desaparecen.
  2. Perfiles de proyección: para cada fila y, cuánta "línea horizontal" hay
     (y lo mismo por columnas para las verticales).
  3. Para cada n candidato (3..9) las líneas interiores deberían estar en
     k·S/n. Puntaje(n) = intensidad media del perfil en esas posiciones menos
     la intensidad máxima dentro de cada celda (lejos de sus bordes). Si n es
     el correcto hay línea en las primeras y papel en el interior; un n
     equivocado falla en una de las dos (con n=9 real, n=3 encuentra líneas
     dentro de sus celdas; con n=3 real, n=9 no encuentra líneas en k·S/9).
  4. Ajuste fino: cada línea se mueve al pico del perfil más cercano.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import binarize, to_gray

N_RANGE = range(3, 10)


def line_masks(rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras binarias con solo las líneas horizontales y verticales largas."""
    gray = to_gray(rect)
    S = gray.shape[0]
    bw = binarize(gray, block=(S // 20) | 1, c=5)
    L = max(15, S // 15)  # más largo que cualquier carácter, más corto que una celda de 9x9
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return horiz, vert


def projection_profiles(rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fracción de cada fila (perfil_y) y columna (perfil_x) cubierta por líneas."""
    horiz, vert = line_masks(rect)
    S = rect.shape[0]
    prof_y = horiz.sum(axis=1) / (255.0 * S)
    prof_x = vert.sum(axis=0) / (255.0 * S)
    return prof_x, prof_y


def _score(profile: np.ndarray, n: int) -> float:
    S = len(profile)
    w = max(2, S // (5 * n))  # tolerancia alrededor de cada posición esperada
    step = S / n
    # "Hay línea donde n dice que debe haberla": máximo cerca de cada k·S/n.
    on = np.mean([profile[max(0, int(k * step) - w): int(k * step) + w + 1].max()
                  for k in range(1, n)])
    # "No hay línea dentro de las celdas": máximo en el interior de cada celda.
    off = np.mean([profile[int(k * step) + w: int((k + 1) * step) - w].max()
                   for k in range(n)])
    return on - off


def detect_n(rect: np.ndarray, n_range=N_RANGE) -> tuple[int, dict[int, float]]:
    """Devuelve (n estimado, puntaje de cada candidato)."""
    prof_x, prof_y = projection_profiles(rect)
    scores = {n: (_score(prof_x, n) + _score(prof_y, n)) / 2 for n in n_range}
    return max(scores, key=scores.get), scores


def refine_lines(profile: np.ndarray, n: int) -> np.ndarray:
    """Posiciones de las n+1 líneas: cada una se ajusta al pico de perfil más cercano."""
    S = len(profile)
    w = max(2, S // (6 * n))
    lines = []
    for k in range(n + 1):
        guess = int(round(k * (S - 1) / n))
        a, b = max(0, guess - w), min(S, guess + w + 1)
        window = profile[a:b]
        if window.max() > 0.3:
            # Centro del tramo con valor máximo (las líneas gruesas son mesetas).
            idx = np.flatnonzero(window >= window.max() * 0.9)
            lines.append(a + (idx[0] + idx[-1]) / 2)
        else:
            lines.append(float(guess))
    return np.array(lines)


def detect_grid(rect: np.ndarray, n: int | None = None):
    """Devuelve (n, xs, ys): n y las coordenadas x / y de las n+1 líneas de la grilla."""
    if n is None:
        n, _ = detect_n(rect)
    prof_x, prof_y = projection_profiles(rect)
    return n, refine_lines(prof_x, n), refine_lines(prof_y, n)
