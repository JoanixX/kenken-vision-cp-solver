"""Pruebas del hito 3: tablero, n, grilla y jaulas sobre imágenes sintéticas."""

import random

import numpy as np
import pytest

from kenken.cages import UnionFind, otsu_1d
from kenken.evaluate import structure_metrics
from kenken.generator import generate
from kenken.pipeline import extract_structure
from kenken.preprocessing import order_corners
from kenken.render import render_sample


@pytest.mark.parametrize("n, photo, level", [(3, False, "normal"), (5, True, "normal"),
                                             (7, True, "normal"), (9, False, "normal"),
                                             (6, True, "hard")])
def test_structure_on_synthetic(n, photo, level):
    inst, sol = generate(n, seed=10 + n)
    img, gt = render_sample(inst, sol, random.Random(n), photo=photo, level=level)
    m = structure_metrics(gt, extract_structure(img))
    assert m["board_err"] < 0.02
    assert m["n_ok"]
    assert m["cages_ok"]


def test_order_corners():
    pts = np.float32([[10, 90], [90, 85], [12, 8], [95, 12]])
    assert order_corners(pts).tolist() == [[12, 8], [95, 12], [90, 85], [10, 90]]


def test_otsu_1d_separates_two_groups():
    thr = otsu_1d([1.0, 1.2, 0.9, 1.1, 6.0, 6.5, 5.8])
    assert 1.2 < thr < 5.8


def test_union_find_groups():
    uf = UnionFind(range(5))
    uf.union(0, 1), uf.union(3, 4), uf.union(1, 4)
    assert sorted(sorted(g) for g in uf.groups()) == [[0, 1, 3, 4], [2]]
