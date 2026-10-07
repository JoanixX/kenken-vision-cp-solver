"""Render sintético de tableros KenKen: imágenes con su ground truth.

Cada imagen se construye en dos etapas:

  1. draw_board():   dibuja el tablero "limpio" visto de frente (como una
                     captura de pantalla), con un estilo aleatorio: tamaño de
                     celda, grosor de líneas fina/gruesa, fuente y glifos de
                     operación (× o x, ÷ o /, − o -).
  2. photograph():   simula una foto: lo pega sobre un fondo con una
                     homografía aleatoria (perspectiva + rotación + escala),
                     y agrega iluminación desigual, sombra, desenfoque, ruido
                     y compresión JPEG.

El ground truth (JSON) contiene la instancia, la solución, las 4 esquinas del
tablero en la imagen, la homografía y la caja de cada etiqueta. Con eso se
puede evaluar cada etapa de la visión por separado y recortar etiquetas para
entrenar/evaluar la CNN.

Uso:  python -m kenken.render --count 300 --out dataset/synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .generator import generate
from .instance import Instance

# ============================================================ fuentes
# Nombres de archivo en Windows / Linux (Colab) / Mac. Se usan las que existan.
FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf", "verdana.ttf",
    "tahoma.ttf", "times.ttf", "timesbd.ttf", "georgia.ttf", "trebuc.ttf",
    "segoeui.ttf", "cour.ttf", "consola.ttf", "comic.ttf", "cambriab.ttf",
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf",
    "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSerif-Regular.ttf",
    "LiberationMono-Regular.ttf", "Arial.ttf", "Helvetica.ttc", "Times New Roman.ttf",
]
FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
             Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
             Path("/Library/Fonts"), Path("/System/Library/Fonts"),
             Path.home() / ".fonts"]

_font_cache: list[str] | None = None


def available_fonts() -> list[str]:
    """Rutas de las fuentes candidatas instaladas en el sistema."""
    global _font_cache
    if _font_cache is None:
        wanted = {name.lower() for name in FONT_CANDIDATES}
        found = {}
        for d in FONT_DIRS:
            if d.is_dir():
                for p in d.rglob("*"):
                    if p.name.lower() in wanted and p.name.lower() not in found:
                        found[p.name.lower()] = str(p)
        _font_cache = sorted(found.values())
        if not _font_cache:
            raise RuntimeError("No se encontró ninguna fuente TrueType conocida "
                               "(en Colab/Linux: apt install fonts-dejavu)")
    return _font_cache


def _has_glyph(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    """¿La fuente tiene el carácter? (si no, PIL dibuja el mismo cuadro 'notdef')."""
    def mask(c):
        return np.array(font.getmask(c))
    m = mask(ch)
    return m.size > 0 and m.any() and not np.array_equal(m, mask("\uffff"))


# ============================================================ estilo
OP_GLYPHS = {"+": ["+"], "-": ["−", "-"], "*": ["×", "x"], "/": ["÷", "/"], "=": [""]}


@dataclass
class Style:
    cell_px: int            # lado de cada celda en píxeles
    thin: int               # grosor de las líneas entre celdas de la misma jaula
    thick: int              # grosor de los bordes de jaula y del contorno
    font_path: str
    font_frac: float        # alto de la etiqueta / lado de la celda
    label_pad: float        # separación de la etiqueta al borde (fracción de celda)
    ink: int                # nivel de gris de la tinta (0 = negro)
    thin_ink: int           # algunos estilos dibujan las líneas finas en gris
    op_glyphs: dict         # operación -> glifo usado en esta imagen


def random_style(rng: random.Random) -> Style:
    cell = rng.randint(60, 120)
    thin = rng.randint(1, max(1, cell // 40))
    thick = rng.randint(max(thin + 2, cell // 30), max(thin + 3, cell // 14))
    font_path = rng.choice(available_fonts())
    font = ImageFont.truetype(font_path, 40)
    glyphs = {}
    for op, opts in OP_GLYPHS.items():
        usable = [g for g in opts if not g or _has_glyph(font, g)]
        glyphs[op] = rng.choice(usable or [opts[-1]])
    ink = rng.randint(0, 50)
    thin_ink = ink if rng.random() < 0.6 else rng.randint(90, 170)
    return Style(cell, thin, thick, font_path, rng.uniform(0.18, 0.28),
                 rng.uniform(0.04, 0.09), ink, thin_ink, glyphs)


# ============================================================ tablero limpio
def draw_board(inst: Instance, style: Style, paper: int = 255):
    """Dibuja el tablero de frente.

    Devuelve (imagen en gris uint8, esquinas del tablero, cajas de etiquetas).
    Las esquinas van en orden TL, TR, BR, BL (convención de todo el proyecto).
    """
    n, c, s = inst.n, style.cell_px, style
    margin = s.thick * 2 + c // 4
    side = n * c
    size = side + 2 * margin
    img = Image.new("L", (size, size), paper)
    draw = ImageDraw.Draw(img)

    def px(k):  # coordenada en píxeles de la línea k de la grilla
        return margin + k * c

    # 1) Líneas finas en toda la grilla interior.
    for k in range(1, n):
        draw.line([(px(k), px(0)), (px(k), px(n))], fill=s.thin_ink, width=s.thin)
        draw.line([(px(0), px(k)), (px(n), px(k))], fill=s.thin_ink, width=s.thin)

    # 2) Segmentos gruesos donde dos celdas vecinas están en jaulas distintas.
    owner = inst.cage_of()
    h = s.thick / 2
    for i in range(n):
        for j in range(n):
            if j + 1 < n and owner[(i, j)] != owner[(i, j + 1)]:   # borde vertical
                x = px(j + 1)
                draw.rectangle([x - h, px(i) - h, x + h - 1, px(i + 1) + h - 1], fill=s.ink)
            if i + 1 < n and owner[(i, j)] != owner[(i + 1, j)]:   # borde horizontal
                y = px(i + 1)
                draw.rectangle([px(j) - h, y - h, px(j + 1) + h - 1, y + h - 1], fill=s.ink)

    # 3) Contorno exterior grueso.
    draw.rectangle([px(0) - h, px(0) - h, px(n) + h - 1, px(n) + h - 1],
                   outline=s.ink, width=s.thick)

    # 4) Etiquetas en la esquina superior izquierda de la celda ancla de cada jaula.
    font = ImageFont.truetype(s.font_path, max(8, int(s.font_frac * c)))
    boxes = []
    for cage in inst.cages:
        i, j = cage.anchor
        text = f"{cage.target}{s.op_glyphs[cage.op]}"
        pad = h + s.label_pad * c
        draw.text((px(j) + pad, px(i) + pad), text, fill=s.ink, font=font, anchor="lt")
        boxes.append(list(draw.textbbox((px(j) + pad, px(i) + pad), text, font=font, anchor="lt")))

    corners = np.float32([[px(0), px(0)], [px(n), px(0)], [px(n), px(n)], [px(0), px(n)]])
    return np.array(img), corners, boxes


# ============================================================ simular foto
def photograph(board: np.ndarray, corners: np.ndarray, rng: random.Random,
               out_size: int = 900):
    """Pega el tablero sobre un fondo con perspectiva y degradaciones de cámara.

    Devuelve (imagen BGR uint8, esquinas en la imagen final, homografía 3x3).
    """
    nrng = np.random.default_rng(rng.randrange(2 ** 32))
    S = out_size

    # Cuadrilátero destino: un cuadrado rotado, escalado y con las esquinas
    # movidas al azar (= perspectiva leve).
    side = rng.uniform(0.55, 0.85) * S
    ang = np.deg2rad(rng.uniform(-12, 12))
    cx, cy = S / 2 + rng.uniform(-0.06, 0.06) * S, S / 2 + rng.uniform(-0.06, 0.06) * S
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    jitter = nrng.uniform(-0.07, 0.07, (4, 2)) * side
    dst = base @ rot.T + jitter
    # El tablero completo debe quedar dentro de la foto (con 3% de margen).
    lo, hi = 0.03 * S, 0.97 * S
    scale = min(1.0, (hi - lo) / np.ptp(dst, axis=0).max())
    dst *= scale
    cx = np.clip(cx, lo - dst[:, 0].min(), hi - dst[:, 0].max())
    cy = np.clip(cy, lo - dst[:, 1].min(), hi - dst[:, 1].max())
    dst_corners = (dst + [cx, cy]).astype(np.float32)

    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)

    # Hoja de papel (algo más grande que el tablero) sobre una mesa.
    paper_tint = np.array([rng.randint(215, 255), rng.randint(215, 255), rng.randint(210, 250)])
    table = np.array([rng.randint(30, 200) for _ in range(3)], dtype=np.float32)
    bg = np.ones((S, S, 3), np.float32) * table
    bg += nrng.normal(0, rng.uniform(2, 12), (S, S, 1))           # textura de la mesa

    h, w = board.shape
    sheet = np.ones((h, w), np.uint8) * 255
    sheet_mask = cv2.warpPerspective(sheet, H, (S, S), flags=cv2.INTER_LINEAR) / 255.0
    warped = cv2.warpPerspective(board, H, (S, S), flags=cv2.INTER_LINEAR,
                                 borderValue=255).astype(np.float32) / 255.0
    paper = warped[..., None] * paper_tint
    img = bg * (1 - sheet_mask[..., None]) + paper * sheet_mask[..., None]

    # Iluminación: gradiente lineal + sombra elíptica suave.
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    gx, gy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
    light = 1 + gx * (xx - 0.5) + gy * (yy - 0.5)
    if rng.random() < 0.5:
        sx, sy, r = rng.random(), rng.random(), rng.uniform(0.2, 0.5)
        shadow = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * r ** 2)))
        light *= 1 - rng.uniform(0.2, 0.5) * shadow
    img *= light[..., None] * rng.uniform(0.75, 1.1)

    # Desenfoque (foco / movimiento) y ruido de sensor.
    k = rng.choice([0, 0, 3, 5])
    if k:
        img = cv2.GaussianBlur(img, (k, k), 0)
    img += nrng.normal(0, rng.uniform(1, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Compresión JPEG.
    q = rng.randint(40, 95)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img, dst_corners, H


def screenshot(board: np.ndarray, corners: np.ndarray, rng: random.Random):
    """Variante 'captura digital': sin perspectiva, solo un margen y algo de ruido JPEG."""
    pad = rng.randint(0, 40)
    img = cv2.copyMakeBorder(board, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if rng.random() < 0.5:
        q = rng.randint(70, 95)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    H = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], np.float64)
    return img, corners + pad, H


# ============================================================ muestra completa
def render_sample(inst: Instance, solution, rng: random.Random, photo: bool):
    """Imagen + ground truth (dict serializable a JSON) de una instancia."""
    style = random_style(rng)
    board, corners, boxes = draw_board(inst, style)
    if photo:
        # Tableros grandes en una imagen más grande, para que las etiquetas sigan legibles.
        img, img_corners, H = photograph(board, corners, rng, out_size=max(900, 130 * inst.n))
    else:
        img, img_corners, H = screenshot(board, corners, rng)

    # Cajas de etiquetas normalizadas al tablero: (0,0) = esquina TL, (1,1) = BR.
    # Así sirven para cualquier rectificación posterior del tablero.
    x0, y0 = map(float, corners[0])
    side = float(corners[1][0] - corners[0][0])
    norm_boxes = [[round((bx0 - x0) / side, 5), round((by0 - y0) / side, 5),
                   round((bx1 - x0) / side, 5), round((by1 - y0) / side, 5)]
                  for bx0, by0, bx1, by1 in boxes]

    gt = inst.to_dict()
    gt["solution"] = solution
    gt["image"] = {
        "kind": "photo" if photo else "screenshot",
        "corners": np.round(img_corners, 2).tolist(),     # TL, TR, BR, BL
        "H": np.asarray(H).tolist(),                       # tablero limpio -> imagen
        "label_boxes": norm_boxes,                         # una por jaula, mismo orden
        "labels": [f"{c.target}{style.op_glyphs[c.op]}" for c in inst.cages],
        "style": {**asdict(style), "font": Path(style.font_path).name},
    }
    del gt["image"]["style"]["font_path"]
    return img, gt


def make_dataset(out_dir: str | Path, count: int, n_range=(3, 9), photo_prob: float = 0.7,
                 seed: int = 0, prefix: str = "syn") -> list[Path]:
    """Genera `count` pares imagen.png + imagen.json en out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for k in range(count):
        n = rng.randint(*n_range)
        inst, solution = generate(n, seed=rng.randrange(2 ** 31))
        img, gt = render_sample(inst, solution, rng, photo=rng.random() < photo_prob)
        stem = out / f"{prefix}_{k:05d}"
        # Las "fotos" ya pasaron por JPEG: guardarlas como .jpg ocupa ~10x menos.
        img_path = stem.with_suffix(".jpg" if gt["image"]["kind"] == "photo" else ".png")
        gt["image"]["file"] = img_path.name
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if img_path.suffix == ".jpg" else []
        cv2.imwrite(str(img_path), img, params)
        with open(stem.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False)
        paths.append(img_path)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera el dataset sintético de KenKen.")
    ap.add_argument("--out", default="dataset/synthetic")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--nmin", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=9)
    ap.add_argument("--photo-prob", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    paths = make_dataset(a.out, a.count, (a.nmin, a.nmax), a.photo_prob, a.seed)
    print(f"{len(paths)} imágenes en {a.out}")
