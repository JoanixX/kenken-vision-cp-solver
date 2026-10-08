"""Pruebas del hito 5: de la imagen a la solución y su visualización."""

import random

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from kenken.cnn import MODEL_PATH  # noqa: E402
from kenken.generator import generate  # noqa: E402
from kenken.instance import Instance  # noqa: E402
from kenken.pipeline import extract_structure, solve_image  # noqa: E402
from kenken.render import render_sample  # noqa: E402
from kenken.visualize import draw_puzzle, overlay_solution, plot_result  # noqa: E402

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="modelo no entrenado")


@needs_model
@pytest.mark.parametrize("n, photo", [(4, False), (5, True), (6, True)])
def test_solve_image_end_to_end(n, photo):
    inst, sol = generate(n, seed=40 + n)
    img, _ = render_sample(inst, sol, random.Random(n), photo=photo)
    r = solve_image(img)
    assert r.status == "solved"
    assert r.solution == sol
    assert r.overlay.shape == img.shape
    assert set(r.times) == {"structure", "ocr", "solve", "render"}


def test_no_board_is_reported():
    blank = np.full((400, 400, 3), 255, np.uint8)
    r = solve_image(blank)
    assert r.status == "no_board" and not r.solved


def test_overlay_changes_only_inside_board():
    inst, sol = generate(4, seed=1)
    img, _ = render_sample(inst, sol, random.Random(3), photo=True)
    st = extract_structure(img)
    out = overlay_solution(img, st, sol)
    diff = np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 30
    ys, xs = np.nonzero(diff)
    assert diff.any()
    c = st.corners
    assert xs.min() >= c[:, 0].min() - 2 and xs.max() <= c[:, 0].max() + 2
    assert ys.min() >= c[:, 1].min() - 2 and ys.max() <= c[:, 1].max() + 2


@needs_model
def test_figures_render(tmp_path):
    inst = Instance.load("examples/4x4_a.json")
    draw_puzzle(inst, [[1, 2, 3, 4]] * 4)
    sol_inst, sol = generate(4, seed=2)
    img, _ = render_sample(sol_inst, sol, random.Random(0), photo=False)
    plot_result(img, solve_image(img), tmp_path / "r.png")
    assert (tmp_path / "r.png").stat().st_size > 10_000
