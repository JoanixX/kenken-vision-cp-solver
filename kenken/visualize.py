"""Visualizaciones: etapas de la visión (hito 3); grilla limpia y solución sobre la foto (hito 5)."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def plot_structure(img: np.ndarray, st, title: str | None = None, axes=None):
    """Tres paneles: tablero detectado, tablero rectificado con la grilla, jaulas encontradas."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
    a0, a1, a2 = axes

    a0.imshow(_rgb(img))
    q = np.vstack([st.corners, st.corners[:1]])
    a0.plot(q[:, 0], q[:, 1], "r-", lw=2)
    a0.set_title("1. Tablero detectado")

    a1.imshow(_rgb(st.rect))
    for x in st.xs:
        a1.axvline(x, color="tab:blue", lw=1)
    for y in st.ys:
        a1.axhline(y, color="tab:blue", lw=1)
    a1.set_title(f"2. Rectificado, n = {st.n}")

    # Cada jaula con un color; los bordes gruesos detectados en rojo.
    rng = np.random.default_rng(0)
    overlay = _rgb(st.rect).copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    tint = overlay.copy()
    for cage in st.cages:
        color = rng.integers(80, 255, 3)
        for i, j in cage:
            tint[int(st.ys[i]):int(st.ys[i + 1]), int(st.xs[j]):int(st.xs[j + 1])] = color
    a2.imshow((0.55 * overlay + 0.45 * tint).astype(np.uint8))
    thr = st.debug.get("threshold")
    for ((i, j), (i2, j2)), mass in st.debug.get("inner", {}).items():
        if thr is not None and mass >= thr:
            if j2 == j + 1:   # borde vertical
                a2.plot([st.xs[j2]] * 2, [st.ys[i], st.ys[i + 1]], "r-", lw=3)
            else:             # borde horizontal
                a2.plot([st.xs[j], st.xs[j + 1]], [st.ys[i2]] * 2, "r-", lw=3)
    a2.set_title(f"3. Jaulas: {len(st.cages)}")

    for a in axes:
        a.axis("off")
    if title:
        a0.figure.suptitle(title)
    return axes
