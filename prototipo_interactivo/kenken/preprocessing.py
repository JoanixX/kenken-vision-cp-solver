"""Preprocesamiento: carga, binarización, detección del tablero y corrección de perspectiva.

Pasos (ver PLAN.md, Fase 1):
  1. Gris + desenfoque gaussiano + umbral adaptativo (robusto a luz desigual).
  2. Contornos -> cuadriláteros candidatos (approxPolyDP con 4 vértices).
  3. Se elige el cuadrilátero más grande cuyo borde sea una LÍNEA OSCURA
     (más oscura que ambos lados). Así se descarta el borde de la hoja de
     papel, que es un escalón de brillo y no una línea.
  4. Homografía (getPerspectiveTransform) y warpPerspective a un cuadrado fijo.

Convención de esquinas en todo el proyecto: TL, TR, BR, BL.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RECT_SIZE = 900  # lado del tablero rectificado, en píxeles


# ============================================================ carga y binarización
def load_image(path: str | Path) -> np.ndarray:
    """Lee una imagen BGR. (cv2.imread falla con tildes en rutas de Windows.)"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"no se pudo leer la imagen: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray, method: str = "adaptive", block: int | None = None,
             c: int = 7) -> np.ndarray:
    """Imagen binaria con la tinta en 255 y el papel en 0.

    adaptive: el umbral de cada píxel es la media gaussiana de su vecindario
              menos c, así una sombra o un gradiente de luz no 'apagan' líneas.
    otsu:     un único umbral global elegido por Otsu (alternativa / baseline).
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if method == "otsu":
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return bw
    if block is None:
        block = max(15, (min(gray.shape) // 25) | 1)  # impar, proporcional a la imagen
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, c)


# ============================================================ tablero
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como TL, TR, BR, BL (por suma y diferencia de coordenadas)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def _edge_contrast(gray: np.ndarray, quad: np.ndarray) -> float:
    """¿El contorno del cuadrilátero es una línea oscura?

    Muestrea puntos a lo largo de los 4 lados; para cada punto compara el gris
    sobre la línea (mínimo en una pequeña franja hacia adentro) con el gris a
    una distancia d hacia adentro y hacia afuera. Devuelve la mediana de
    min(adentro, afuera) - línea: alto para un borde de tablero, ~0 para el
    borde de una hoja.
    """
    h, w = gray.shape
    center = quad.mean(axis=0)
    side = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1).mean()
    d = max(6.0, 0.03 * side)
    t = np.linspace(0.1, 0.9, 40)
    vals = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        pts = a + t[:, None] * (b - a)
        inward = center - pts
        inward /= np.linalg.norm(inward, axis=1, keepdims=True)

        def sample(offset):
            p = np.round(pts + offset * inward).astype(int)
            p[:, 0] = p[:, 0].clip(0, w - 1)
            p[:, 1] = p[:, 1].clip(0, h - 1)
            return gray[p[:, 1], p[:, 0]].astype(np.float32)

        line = np.min([sample(o) for o in np.arange(0, d / 2, 1.0)], axis=0)
        vals.append(np.minimum(sample(d), sample(-d)) - line)
    return float(np.median(np.concatenate(vals)))


def find_board(img: np.ndarray, min_contrast: float = 25.0, work_size: int = 1000) -> np.ndarray:
    """Encuentra las 4 esquinas del tablero (TL, TR, BR, BL) en coordenadas de `img`."""
    gray = to_gray(img)
    scale = work_size / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else gray
    scale = min(scale, 1.0)
    bw = binarize(small)
    # Cerrar pequeñas interrupciones del contorno (líneas finas, JPEG, desenfoque).
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.05 * small.shape[0] * small.shape[1]
    blur = cv2.GaussianBlur(small, (3, 3), 0)

    best, fallback = None, None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_corners(approx)
        if fallback is None:
            fallback = quad
        if _edge_contrast(blur, quad) >= min_contrast:
            best = quad
            break
    if best is None:
        if fallback is None:
            raise ValueError("no se encontró ningún cuadrilátero que parezca un tablero")
        best = fallback
    return best / scale


def rectify(img: np.ndarray, corners: np.ndarray, size: int = RECT_SIZE):
    """Corrige la perspectiva: lleva el cuadrilátero a un cuadrado size x size.

    Devuelve (imagen rectificada, H) con H: imagen original -> rectificada.
    La inversa de H sirve luego para dibujar la solución sobre la foto.
    """
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    H = cv2.getPerspectiveTransform(np.float32(corners), dst)
    return cv2.warpPerspective(img, H, (size, size), flags=cv2.INTER_LINEAR), H
