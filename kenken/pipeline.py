"""Pipeline de visión: imagen -> estructura del tablero (y, más adelante, -> solución).

Hito 3: tablero, perspectiva, n, grilla y jaulas.
Hito 5: se agregan OCR de etiquetas, modelo CP y visualización (solve_image).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .cages import detect_cages
from .grid import detect_grid
from .preprocessing import RECT_SIZE, find_board, rectify


@dataclass
class Structure:
    corners: np.ndarray        # 4 esquinas del tablero en la imagen original (TL, TR, BR, BL)
    H: np.ndarray              # homografía imagen original -> tablero rectificado
    rect: np.ndarray           # tablero rectificado (RECT_SIZE x RECT_SIZE)
    n: int
    xs: np.ndarray             # n+1 posiciones x de las líneas en `rect`
    ys: np.ndarray             # n+1 posiciones y de las líneas en `rect`
    cages: list[list[tuple[int, int]]]
    debug: dict = field(default_factory=dict)


def extract_structure(img: np.ndarray, size: int = RECT_SIZE, use_labels: bool = True) -> Structure:
    """Imagen -> tablero, perspectiva, n, grilla y jaulas (sin leer las etiquetas todavía)."""
    corners = find_board(img)
    rect, H = rectify(img, corners, size)
    n, xs, ys = detect_grid(rect)
    cages, info = detect_cages(rect, xs, ys, use_labels=use_labels)
    # El contorno detectado es el borde EXTERIOR de la línea gruesa; las líneas
    # de la grilla dan su centro. Se proyectan de vuelta a la imagen original.
    grid_corners = np.float32([[xs[0], ys[0]], [xs[-1], ys[0]], [xs[-1], ys[-1]], [xs[0], ys[-1]]])
    corners = cv2.perspectiveTransform(grid_corners[None], np.linalg.inv(H))[0]
    info["contour_corners"] = corners
    return Structure(corners, H, rect, n, xs, ys, cages, info)
