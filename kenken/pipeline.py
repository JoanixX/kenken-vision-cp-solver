"""Pipeline de visión: imagen -> estructura del tablero (y, más adelante, -> solución).

Hito 3: tablero, perspectiva, n, grilla y jaulas      -> extract_structure()
Hito 4: lectura de etiquetas con la CNN                -> read_instance()
Hito 5: modelo CP y visualización                      -> solve_image()
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cages import detect_cages
from .grid import detect_grid
from .instance import Cage, Instance, InstanceError
from .model import SolveResult, solve
from .ocr import allowed_ops, decode_readings, extract_label_glyphs
from .preprocessing import RECT_SIZE, find_board, load_image, rectify


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


def read_instance(st: Structure, k: int = 5, model=None, alternatives: bool = True) -> Instance:
    """Lee la etiqueta de cada jaula con la CNN y arma la instancia.

    Para cada jaula se prueban la segmentación principal y sus alternativas
    (ver ocr.segmentation_hypotheses); cada una se decodifica con la gramática
    y se juntan todas las lecturas (restando la penalización de la
    alternativa). La jaula queda con la lectura más probable, y en
    `candidates[i]` se guardan las k mejores con su log-probabilidad: las usa
    la inferencia conjunta del modelo CP. Si una etiqueta no se puede leer,
    su lista queda vacía y se anota en debug["unread"].
    """
    from .cnn import predict_log_probs  # import local: torch solo se carga si se usa

    hyp_sets = extract_label_glyphs(st.rect, st.xs, st.ys, st.cages, alternatives=alternatives)
    flat = [g for hyps in hyp_sets for g, _ in hyps]
    sizes = [len(g) for g in flat]
    all_glyphs = np.concatenate(flat) if sum(sizes) else np.zeros((0, 32, 32), np.float32)
    log_probs = predict_log_probs(all_glyphs, model)   # una sola pasada por la CNN
    splits = iter(np.split(log_probs, np.cumsum(sizes)[:-1]))

    cages, candidates, unread, main_lp = [], {}, [], []
    for idx, (cells, hyps) in enumerate(zip(st.cages, hyp_sets)):
        best: dict = {}
        for h, (_, penalty) in enumerate(hyps):
            lp = next(splits)
            if h == 0:
                main_lp.append(lp)
            for r in decode_readings(lp, len(cells), st.n, k=k) if len(lp) else []:
                key = (r["target"], r["op"])
                score = r["logp"] - penalty
                if key not in best or score > best[key]["logp"]:
                    best[key] = {**r, "logp": score}
        readings = sorted(best.values(), key=lambda r: -r["logp"])[:k]
        candidates[idx] = readings
        if readings:
            cages.append(Cage(cells, readings[0]["target"], readings[0]["op"]))
        else:
            unread.append(idx)
            cages.append(Cage(cells, 0, allowed_ops(len(cells))[0]))
    st.debug["unread"] = unread
    st.debug["glyph_log_probs"] = main_lp
    return Instance(st.n, cages, candidates)


# ============================================================ hito 5: de la foto a la solución
@dataclass
class ImageResult:
    status: str                        # solved / infeasible / invalid_instance / no_board / ...
    structure: Structure | None = None
    instance: Instance | None = None
    solve: SolveResult | None = None
    solution: list | None = None
    overlay: np.ndarray | None = None  # foto original con la solución dibujada
    corrected: set = field(default_factory=set)   # jaulas cuya lectura cambió el solver (hito 6)
    times: dict = field(default_factory=dict)     # segundos por etapa
    message: str = ""

    @property
    def solved(self) -> bool:
        return self.solution is not None


def solve_image(image, k: int = 5, time_limit: float = 30.0) -> ImageResult:
    """Imagen (ruta o arreglo BGR) -> solución, sin intervención manual.

    Etapas: tablero y jaulas (visión clásica) -> etiquetas (CNN) -> instancia
    validada -> modelo CP-SAT -> solución dibujada sobre la foto original.
    """
    from .visualize import overlay_solution  # import local: matplotlib solo si se usa

    img = load_image(image) if isinstance(image, (str, Path)) else image
    times = {}

    t = time.perf_counter()
    try:
        st = extract_structure(img)
    except ValueError as e:
        return ImageResult("no_board", message=str(e), times={"structure": time.perf_counter() - t})
    times["structure"] = time.perf_counter() - t

    t = time.perf_counter()
    inst = read_instance(st, k=k)
    times["ocr"] = time.perf_counter() - t
    res = ImageResult("read", structure=st, instance=inst, times=times)

    try:
        inst.validate()
    except InstanceError as e:
        res.status, res.message = "invalid_instance", str(e)
        return res

    t = time.perf_counter()
    res.solve = solve(inst, time_limit=time_limit)
    times["solve"] = time.perf_counter() - t
    if not res.solve.solved:
        res.status = res.solve.status.lower()  # 'infeasible' (o 'unknown' si se acabó el tiempo)
        res.message = "las lecturas más probables no tienen solución"
        return res

    t = time.perf_counter()
    res.solution = res.solve.grid
    res.overlay = overlay_solution(img, st, res.solution)
    times["render"] = time.perf_counter() - t
    res.status = "solved"
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Resuelve un KenKen desde una foto.")
    ap.add_argument("image")
    ap.add_argument("--out", help="figura resumen (png); por defecto <imagen>_solucion.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from .model import format_grid
    from .visualize import plot_result

    r = solve_image(a.image)
    print(f"estado: {r.status}  {r.message}")
    if r.instance is not None:
        print(r.instance)
    if r.solved:
        print(format_grid(r.solution))
    print("tiempos: " + ", ".join(f"{k}={v:.3f}s" for k, v in r.times.items()))
    out = a.out or str(Path(a.image).with_name(Path(a.image).stem + "_solucion.png"))
    plot_result(load_image(a.image), r, out)
    print("figura:", out)
