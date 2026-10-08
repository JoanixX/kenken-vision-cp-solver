"""Visualizaciones: etapas de la visión y renderizado de la solución como imagen.

Provee:
  - plot_structure(): 3 paneles para depuración de estructura (detección, rectificado, jaulas).
  - draw_solution_clean(): grilla vectorial limpia con jaulas y números resueltos.
  - overlay_solution_on_rectified(): dígitos superpuestos sobre el tablero rectificado.
  - overlay_solution_on_original(): dígitos proyectados sobre la imagen original (con perspectiva).
  - render_solution_composite(): comparativa lado a lado (Entrada original / Solución KenKen).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .instance import Instance
    from .pipeline import Structure

OP_SYMBOLS = {
    "+": "+",
    "-": "−",
    "*": "×",
    "/": "÷",
    "=": "",
}


def _rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Obtiene una fuente TrueType legible en Windows, Linux o Mac."""
    candidates = (
        ["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass

    try:
        from .render import available_fonts

        for p in available_fonts():
            if ("bd" in p.lower() or "bold" in p.lower()) == bold:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


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
            if j2 == j + 1:  # borde vertical
                a2.plot([st.xs[j2]] * 2, [st.ys[i], st.ys[i + 1]], "r-", lw=3)
            else:  # borde horizontal
                a2.plot([st.xs[j], st.xs[j + 1]], [st.ys[i2]] * 2, "r-", lw=3)
    a2.set_title(f"3. Jaulas: {len(st.cages)}")

    for a in axes:
        a.axis("off")
    if title:
        a0.figure.suptitle(title)
    return axes


def draw_solution_clean(
    inst: Instance,
    grid: list[list[int]],
    cell_size: int = 120,
    out_path: str | Path | None = None,
    title: str | None = None,
) -> np.ndarray:
    """Genera una imagen gráfica limpia y nítida del tablero KenKen resuelto.

    Muestra:
      - Bordes de jaula gruesos y bordes internos finos.
      - Etiquetas aritméticas en la esquina superior izquierda de cada jaula.
      - Dígitos de la solución centrados en tipografía azul destacada.
    """
    n = inst.n
    margin = 30
    header_h = 60 if title is not None or title != "" else 20
    if title is None:
        title = f"KenKen {n}×{n} — Solución"

    total_w = n * cell_size + 2 * margin
    total_h = n * cell_size + 2 * margin + (header_h if title else 0)

    # Imagen base blanca
    pil_img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)

    # Encabezado
    if title:
        title_font = _get_font(26, bold=True)
        sub_font = _get_font(14, bold=False)
        draw.text((margin, 12), title, fill=(20, 30, 60), font=title_font)
        draw.text((margin, 42), "Resuelto con Google OR-Tools CP-SAT", fill=(100, 110, 130), font=sub_font)

    top_y = margin + (header_h if title else 0)

    # Mapeo de celda a índice de jaula
    cell_to_cage: dict[tuple[int, int], int] = {}
    for c_idx, cage in enumerate(inst.cages):
        for cell in cage.cells:
            cell_to_cage[tuple(cell)] = c_idx

    # 1. Fondo suave y líneas interiores finas
    thin_color = (215, 220, 228)
    for r in range(n):
        for c in range(n):
            x0 = margin + c * cell_size
            y0 = top_y + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            # Fondo celda
            draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

            # Líneas delgadas interiores si pertenecen a la misma jaula
            if c < n - 1 and cell_to_cage.get((r, c)) == cell_to_cage.get((r, c + 1)):
                draw.line([(x1, y0), (x1, y1)], fill=thin_color, width=1)
            if r < n - 1 and cell_to_cage.get((r, c)) == cell_to_cage.get((r + 1, c)):
                draw.line([(x0, y1), (x1, y1)], fill=thin_color, width=1)

    # 2. Bordes gruesos de jaula
    thick_color = (25, 25, 30)
    thick_w = 4
    for r in range(n):
        for c in range(n):
            x0 = margin + c * cell_size
            y0 = top_y + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            curr_c = cell_to_cage.get((r, c))

            # Borde derecho
            if c == n - 1 or curr_c != cell_to_cage.get((r, c + 1)):
                draw.line([(x1, y0), (x1, y1)], fill=thick_color, width=thick_w)
            # Borde inferior
            if r == n - 1 or curr_c != cell_to_cage.get((r + 1, c)):
                draw.line([(x0, y1), (x1, y1)], fill=thick_color, width=thick_w)
            # Borde superior
            if r == 0 or curr_c != cell_to_cage.get((r - 1, c)):
                draw.line([(x0, y0), (x1, y0)], fill=thick_color, width=thick_w)
            # Borde izquierdo
            if c == 0 or curr_c != cell_to_cage.get((r, c - 1)):
                draw.line([(x0, y0), (x0, y1)], fill=thick_color, width=thick_w)

    # Marco exterior reforzado
    draw.rectangle(
        [margin, top_y, margin + n * cell_size, top_y + n * cell_size],
        outline=thick_color,
        width=thick_w + 1,
    )

    # 3. Etiquetas de cada jaula (esquina superior izquierda de la primera celda)
    label_font = _get_font(max(14, int(cell_size * 0.20)), bold=True)
    for cage in inst.cages:
        tl_cell = min(cage.cells, key=lambda p: (p[0], p[1]))
        r, c = tl_cell
        op_sym = OP_SYMBOLS.get(cage.op, cage.op)
        label_text = f"{cage.target}{op_sym}"
        lx = margin + c * cell_size + 6
        ly = top_y + r * cell_size + 4
        draw.text((lx, ly), label_text, fill=(35, 35, 45), font=label_font)

    # 4. Dígitos de la solución centrados
    digit_font = _get_font(max(24, int(cell_size * 0.48)), bold=True)
    digit_color = (13, 71, 161)  # Azul real intenso
    for r in range(n):
        for c in range(n):
            val_str = str(grid[r][c])
            cx = margin + c * cell_size + cell_size / 2
            cy = top_y + r * cell_size + cell_size / 2 + int(cell_size * 0.08)
            draw.text((cx, cy), val_str, fill=digit_color, font=digit_font, anchor="mm")

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), bgr)

    return bgr


def overlay_solution_on_rectified(
    rect_img: np.ndarray,
    xs: np.ndarray | list[float],
    ys: np.ndarray | list[float],
    grid: list[list[int]],
    out_path: str | Path | None = None,
) -> np.ndarray:
    """Superpone los números de la solución centrados sobre el tablero rectificado."""
    n = len(grid)
    canvas = rect_img.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for r in range(n):
        for c in range(n):
            x0, x1 = float(xs[c]), float(xs[c + 1])
            y0, y1 = float(ys[r]), float(ys[r + 1])
            cw, ch = x1 - x0, y1 - y0
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2 + ch * 0.08

            val_str = str(grid[r][c])
            font_size = max(18, int(ch * 0.44))
            font = _get_font(font_size, bold=True)

            # Dibujar resplandor / borde blanco para máxima legibilidad
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)]:
                draw.text((cx + dx, cy + dy), val_str, fill=(255, 255, 255), font=font, anchor="mm")
            draw.text((cx, cy), val_str, fill=(0, 102, 204), font=font, anchor="mm")

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), bgr)
    return bgr


def overlay_solution_on_original(
    orig_img: np.ndarray,
    st: Structure,
    grid: list[list[int]],
    out_path: str | Path | None = None,
) -> np.ndarray:
    """Proyecta los dígitos de la solución sobre la foto/imagen original usando la homografía inversa H⁻¹."""
    n = len(grid)
    canvas = orig_img.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # 1. Contorno del tablero detectado en verde
    if hasattr(st, "corners") and st.corners is not None:
        pts = np.int32(st.corners).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 220, 60), thickness=3)

    # 2. Homografía inversa: rectificado -> imagen original
    inv_H = np.linalg.inv(st.H)

    # Estimar tamaño de celda aproximado en la imagen original
    diag = np.linalg.norm(st.corners[0] - st.corners[2])
    approx_cell = max(20, int(diag / (n * 1.414)))

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for r in range(n):
        for c in range(n):
            cx_r = float(st.xs[c] + st.xs[c + 1]) / 2.0
            cy_r = float(st.ys[r] + st.ys[r + 1]) / 2.0 + float(st.ys[r + 1] - st.ys[r]) * 0.08

            pt_r = np.array([[[cx_r, cy_r]]], dtype=np.float32)
            pt_orig = cv2.perspectiveTransform(pt_r, inv_H)[0][0]
            ox, oy = float(pt_orig[0]), float(pt_orig[1])

            val_str = str(grid[r][c])
            font_size = max(16, int(approx_cell * 0.48))
            font = _get_font(font_size, bold=True)

            # Fondo/halo blanco para contraste
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)]:
                draw.text((ox + dx, oy + dy), val_str, fill=(255, 255, 255), font=font, anchor="mm")
            # Texto azul brillante
            draw.text((ox, oy), val_str, fill=(0, 80, 220), font=font, anchor="mm")

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), bgr)
    return bgr


def render_solution_composite(
    orig_img: np.ndarray,
    inst: Instance,
    grid: list[list[int]],
    st: Structure | None = None,
    out_path: str | Path | None = None,
    title: str | None = None,
) -> np.ndarray:
    """Genera una imagen compuesta lado a lado: Entrada original vs Solución KenKen."""
    clean_board = draw_solution_clean(inst, grid, cell_size=110, title="")

    if st is not None:
        left_panel = overlay_solution_on_original(orig_img, st, grid)
    else:
        left_panel = orig_img.copy()

    # Redimensionar manteniendo altura uniforme
    target_h = 750
    h_left, w_left = left_panel.shape[:2]
    w_left_target = int(w_left * (target_h / h_left))
    left_resized = cv2.resize(left_panel, (w_left_target, target_h), interpolation=cv2.INTER_AREA)

    h_right, w_right = clean_board.shape[:2]
    w_right_target = int(w_right * (target_h / h_right))
    right_resized = cv2.resize(clean_board, (w_right_target, target_h), interpolation=cv2.INTER_AREA)

    banner_h = 80
    total_w = w_left_target + w_right_target + 30
    total_h = target_h + banner_h

    composite = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    # Insertar paneles
    composite[banner_h:banner_h + target_h, 10:10 + w_left_target] = left_resized
    composite[banner_h:banner_h + target_h, 20 + w_left_target:20 + w_left_target + w_right_target] = right_resized

    # Agregar encabezado y subtítulos
    pil_comp = Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_comp)

    banner_font = _get_font(26, bold=True)
    label_font = _get_font(18, bold=True)

    header_text = title or f"KenKen {inst.n}×{inst.n} — Solución Completa End-to-End"
    draw.text((20, 16), header_text, fill=(20, 30, 60), font=banner_font)
    draw.text((20, 48), "Visión Computacional + Constraint Programming (OR-Tools CP-SAT)", fill=(90, 100, 120), font=_get_font(14))

    # Títulos sobre paneles
    draw.text((15, banner_h - 22), "1. Entrada y Detección Proyectada", fill=(50, 60, 80), font=label_font)
    draw.text((25 + w_left_target, banner_h - 22), "2. Tablero y Solución Resuelta", fill=(50, 60, 80), font=label_font)

    bgr = cv2.cvtColor(np.array(pil_comp), cv2.COLOR_RGB2BGR)

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), bgr)

    return bgr
