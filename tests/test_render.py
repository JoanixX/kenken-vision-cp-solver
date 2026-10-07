"""Pruebas del hito 2: render sintético y su ground truth."""

import json
import random

import cv2
import numpy as np
import pytest

from kenken.generator import generate
from kenken.instance import Instance
from kenken.render import draw_board, make_dataset, random_style, render_sample


@pytest.mark.parametrize("photo", [False, True])
def test_corners_match_homography(photo):
    inst, sol = generate(5, seed=3)
    rng = random.Random(0)
    img, gt = render_sample(inst, sol, rng, photo=photo)
    H = np.array(gt["image"]["H"])
    corners = np.array(gt["image"]["corners"])
    h, w = img.shape[:2]
    assert ((corners >= 0) & (corners < [w, h])).all()
    # Rectificar con las esquinas del GT debe dar un tablero oscuro en el contorno.
    side = 400
    M = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                    np.float32([[0, 0], [side, 0], [side, side], [0, side]]))
    rect = cv2.cvtColor(cv2.warpPerspective(img, M, (side, side)), cv2.COLOR_BGR2GRAY)
    assert rect[:, :3].mean() < rect[side // 4: 3 * side // 4, side // 4: 3 * side // 4].mean()
    assert H.shape == (3, 3)


def test_label_boxes_inside_anchor_cells():
    inst, sol = generate(6, seed=4)
    _, gt = render_sample(inst, sol, random.Random(1), photo=False)
    n = inst.n
    for cage, (x0, y0, x1, y1) in zip(inst.cages, gt["image"]["label_boxes"]):
        i, j = cage.anchor
        assert j / n <= x0 < x1 <= (j + 1) / n
        assert i / n <= y0 < y1 <= (i + 1) / n


def test_board_has_thick_cage_borders():
    inst = Instance.load("examples/4x4_a.json")
    style = random_style(random.Random(2))
    img, corners, _ = draw_board(inst, style)
    c, x0, y0 = style.cell_px, int(corners[0][0]), int(corners[0][1])
    # Entre (0,0) y (0,1) hay borde de jaula; entre (0,1) y (0,2) no.
    y = y0 + c // 2 + c // 6
    thick_col = img[y, x0 + c - style.thick: x0 + c + style.thick]
    thin_col = img[y, x0 + 2 * c - style.thick: x0 + 2 * c + style.thick]
    assert (thick_col < 128).sum() > (thin_col < 128).sum()


def test_make_dataset_is_deterministic(tmp_path):
    a = make_dataset(tmp_path / "a", 2, (3, 4), seed=7)
    b = make_dataset(tmp_path / "b", 2, (3, 4), seed=7)
    for pa, pb in zip(a, b):
        assert json.loads(pa.with_suffix(".json").read_text("utf-8")) == \
               json.loads(pb.with_suffix(".json").read_text("utf-8"))
        assert np.array_equal(cv2.imread(str(pa)), cv2.imread(str(pb)))
        gt = json.loads(pa.with_suffix(".json").read_text("utf-8"))
        Instance.from_dict(gt).validate()
