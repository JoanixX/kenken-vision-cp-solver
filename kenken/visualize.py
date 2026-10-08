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

    plot_cages(a2, st)

    for a in axes:
        a.axis("off")
    if title:
        a0.figure.suptitle(title)
    return axes


def plot_cages(ax, st):
    """Tablero rectificado con cada jaula de un color y los bordes gruesos en rojo."""
    rng = np.random.default_rng(0)
    overlay = _rgb(st.rect).copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    tint = overlay.copy()
    for cage in st.cages:
        color = rng.integers(80, 255, 3)
        for i, j in cage:
            tint[int(st.ys[i]):int(st.ys[i + 1]), int(st.xs[j]):int(st.xs[j + 1])] = color
    ax.imshow((0.55 * overlay + 0.45 * tint).astype(np.uint8))
    thr = st.debug.get("threshold")
    for ((i, j), (i2, j2)), mass in st.debug.get("inner", {}).items():
        if thr is not None and mass >= thr:
            if j2 == j + 1:   # borde vertical
                ax.plot([st.xs[j2]] * 2, [st.ys[i], st.ys[i + 1]], "r-", lw=3)
            else:             # borde horizontal
                ax.plot([st.xs[j], st.xs[j + 1]], [st.ys[i2]] * 2, "r-", lw=3)
    ax.set_title(f"3. Jaulas: {len(st.cages)}")
    ax.axis("off")


# ============================================================ hito 5: solución
OP_SYMBOL = {"+": "+", "-": "−", "*": "×", "/": "÷", "=": ""}


def draw_puzzle(inst, solution=None, ax=None, title: str | None = None,
                solution_color: str = "tab:blue", highlight=None):
    """Grilla limpia en matplotlib: líneas finas, bordes de jaula gruesos,
    etiquetas arriba a la izquierda y (opcional) la solución en color.

    highlight: conjunto de índices de jaula a marcar en rojo (p. ej. etiquetas
    corregidas por la inferencia conjunta).
    """
    n = inst.n
    if ax is None:
        _, ax = plt.subplots(figsize=(0.8 * n + 1, 0.8 * n + 1))
    owner = inst.cage_of()
    for k in range(n + 1):
        ax.plot([0, n], [k, k], color="0.75", lw=0.8)
        ax.plot([k, k], [0, n], color="0.75", lw=0.8)
    thick = dict(color="black", lw=3, solid_capstyle="projecting")
    for i in range(n):
        for j in range(n):
            if j + 1 < n and owner[(i, j)] != owner[(i, j + 1)]:
                ax.plot([j + 1, j + 1], [i, i + 1], **thick)
            if i + 1 < n and owner[(i, j)] != owner[(i + 1, j)]:
                ax.plot([j, j + 1], [i + 1, i + 1], **thick)
    ax.plot([0, n, n, 0, 0], [0, 0, n, n, 0], **thick)
    for k, cage in enumerate(inst.cages):
        i, j = cage.anchor
        color = "tab:red" if highlight and k in highlight else "black"
        ax.text(j + 0.07, i + 0.08, f"{cage.target}{OP_SYMBOL[cage.op]}", ha="left", va="top",
                fontsize=max(6, 15 - n), color=color, fontweight="bold" if color != "black" else None)
    if solution is not None:
        for i in range(n):
            for j in range(n):
                ax.text(j + 0.5, i + 0.6, str(solution[i][j]), ha="center", va="center",
                        fontsize=max(10, 30 - 2 * n), color=solution_color)
    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(n + 0.05, -0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def overlay_solution(img: np.ndarray, st, solution, color=(200, 60, 20)) -> np.ndarray:
    """Dibuja la solución sobre la foto ORIGINAL.

    Los dígitos se dibujan en el tablero rectificado (donde las celdas son
    cuadradas) y esa capa se proyecta a la foto con la homografía inversa H⁻¹,
    así quedan con la misma perspectiva que el papel.
    """
    S = st.rect.shape[0]
    layer = np.zeros((S, S, 3), np.uint8)
    alpha = np.zeros((S, S), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(st.n):
        for j in range(st.n):
            cw, ch = st.xs[j + 1] - st.xs[j], st.ys[i + 1] - st.ys[i]
            text = str(solution[i][j])
            scale = 0.016 * ch
            thick = max(1, int(0.045 * ch))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            org = (int(st.xs[j] + (cw - tw) / 2), int(st.ys[i] + 0.62 * ch + th / 2))
            cv2.putText(layer, text, org, font, scale, color, thick, cv2.LINE_AA)
            cv2.putText(alpha, text, org, font, scale, 255, thick, cv2.LINE_AA)
    Hinv = np.linalg.inv(st.H)
    h, w = img.shape[:2]
    layer = cv2.warpPerspective(layer, Hinv, (w, h), flags=cv2.INTER_LINEAR)
    a = cv2.warpPerspective(alpha, Hinv, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    out = img.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a
    return out.astype(np.uint8)


def plot_result(img: np.ndarray, res, path=None):
    """Figura resumen de solve_image: foto con el tablero detectado | jaulas |
    puzzle leído y resuelto | solución sobre la foto original."""
    fig, (a0, a1, a2, a3) = plt.subplots(1, 4, figsize=(20, 5.2))
    a0.imshow(_rgb(img))
    st = res.structure
    if st is not None:
        q = np.vstack([st.corners, st.corners[:1]])
        a0.plot(q[:, 0], q[:, 1], "r-", lw=2)
        plot_cages(a1, st)
        a1.set_title(f"2. Rectificado: n = {st.n}, {len(st.cages)} jaulas")
    a0.set_title("1. Foto y tablero detectado")
    if res.instance is not None:
        draw_puzzle(res.instance, res.solution, ax=a2, highlight=res.corrected,
                    title=f"3. Leído y resuelto: {res.status}")
    a3.imshow(_rgb(res.overlay if res.overlay is not None else img))
    a3.set_title("4. Solución sobre la foto")
    for a in (a0, a1, a2, a3):
        a.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=90)
    return fig
