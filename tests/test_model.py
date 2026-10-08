"""Pruebas del hito 1: instancia JSON, validación, modelo CP y generador.

Ejecutar desde TP/:   python -m pytest -q
"""

from pathlib import Path

import pytest

from kenken.generator import generate
from kenken.instance import Cage, Instance, InstanceError
from kenken.model import (
    cage_satisfied,
    check_solution,
    compute_allowed_tuples,
    find_solutions,
    has_unique_solution,
    solve,
)

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_solve_uniquely(path):
    inst = Instance.load(path).validate()
    res = solve(inst)
    assert res.status == "OPTIMAL"
    assert check_solution(inst, res.grid)
    assert has_unique_solution(inst)


@pytest.mark.parametrize("redundant", [False, True])
def test_redundant_constraint_gives_same_solution(redundant):
    inst = Instance.load(EXAMPLES[1])
    assert check_solution(inst, solve(inst, redundant=redundant).grid)


def test_json_roundtrip(tmp_path):
    inst = Instance.load(EXAMPLES[0])
    inst.save(tmp_path / "x.json")
    assert Instance.load(tmp_path / "x.json").to_dict() == inst.to_dict()


def test_op_aliases_from_ocr():
    assert Cage([[0, 0], [0, 1]], 6, "×").op == "*"
    assert Cage([[0, 0], [0, 1]], 2, "÷").op == "/"
    assert Cage([[0, 0]], 3, "").op == "="


@pytest.mark.parametrize("cells", [[[0, 0], [0, 1]], [[0, 1], [0, 0]]])
def test_division_works_in_both_directions(cells):
    """La división reificada debe aceptar el mayor en cualquiera de las dos celdas."""
    inst = Instance(2, [Cage(cells, 2, "/"), Cage([[1, 0], [1, 1]], 3, "+")])
    sols = find_solutions(inst, limit=10)
    assert sorted(sols) == [[[1, 2], [2, 1]], [[2, 1], [1, 2]]]


def test_infeasible_instance():
    # Fila 0 de un 3x3 sumando 7 es imposible (siempre suma 6).
    inst = Instance(3, [Cage([[0, 0], [0, 1], [0, 2]], 7, "+"),
                        Cage([[1, 0], [1, 1], [1, 2], [2, 2]], 9, "+"),
                        Cage([[2, 0], [2, 1]], 3, "+")])
    assert solve(inst).status == "INFEASIBLE"


@pytest.mark.parametrize("bad, fragment", [
    ({"n": 2, "cages": [{"cells": [[0, 0], [0, 1]], "target": 3, "op": "+"}]}, "sin jaula"),
    ({"n": 2, "cages": [{"cells": [[0, 0], [1, 1]], "target": 3, "op": "+"},
                        {"cells": [[0, 1], [1, 0]], "target": 3, "op": "+"}]}, "no conexas"),
    ({"n": 3, "cages": [{"cells": [[0, 0], [0, 1], [0, 2]], "target": 2, "op": "-"},
                        {"cells": [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
                         "target": 12, "op": "+"}]}, "exactamente 2"),
    ({"n": 4, "cages": [{"cells": [[i, j]], "target": 5, "op": "="}
                        for i in range(4) for j in range(4)]}, "fuera de 1..4"),
])
def test_validation_rejects(bad, fragment):
    with pytest.raises(InstanceError, match=fragment):
        Instance.from_dict(bad).validate()


@pytest.mark.parametrize("n", range(3, 10))
def test_generator_produces_unique_puzzles(n):
    inst, solution = generate(n, seed=100 + n)
    assert check_solution(inst, solution)
    assert find_solutions(inst, limit=2) == [solution]


def test_compute_allowed_tuples_equals():
    c_ok = Cage([(0, 1)], 3, "=")
    assert compute_allowed_tuples(4, c_ok) == [(3,)]
    c_out = Cage([(0, 1)], 5, "=")
    assert compute_allowed_tuples(4, c_out) == []


def test_compute_allowed_tuples_subtraction():
    c_sub = Cage([(0, 0), (0, 1)], 2, "-")
    tuples = compute_allowed_tuples(4, c_sub)
    assert set(tuples) == {(1, 3), (3, 1), (2, 4), (4, 2)}
    for t in tuples:
        assert cage_satisfied(c_sub, list(t))


def test_compute_allowed_tuples_division():
    c_div = Cage([(0, 0), (1, 0)], 2, "/")
    tuples = compute_allowed_tuples(4, c_div)
    assert set(tuples) == {(2, 1), (1, 2), (4, 2), (2, 4)}
    for t in tuples:
        assert cage_satisfied(c_div, list(t))


def test_compute_allowed_tuples_addition_collinear_vs_l_shape():
    # 3 celdas colineares en fila 0: deben ser todas distintas
    c_line = Cage([(0, 0), (0, 1), (0, 2)], 7, "+")
    tuples_line = compute_allowed_tuples(4, c_line)
    assert len(tuples_line) == 6
    for t in tuples_line:
        assert len(set(t)) == 3  # todas distintas
        assert sum(t) == 7

    # 3 celdas en forma de L: celda (0,0) y celda (1,1) NO comparten fila ni columna
    c_l = Cage([(0, 0), (0, 1), (1, 1)], 7, "+")
    tuples_l = compute_allowed_tuples(4, c_l)
    assert len(tuples_l) == 8  # incluye (2, 3, 2) y (3, 1, 3)
    assert (2, 3, 2) in tuples_l
    assert (3, 1, 3) in tuples_l
    for t in tuples_l:
        assert t[0] != t[1]  # celdas (0,0) y (0,1) comparten fila
        assert t[1] != t[2]  # celdas (0,1) y (1,1) comparten columna
        assert sum(t) == 7


def test_compute_allowed_tuples_multiplication():
    # Jaula en L con target 12 en grilla 4x4
    c_prod = Cage([(0, 0), (0, 1), (1, 1)], 12, "*")
    tuples = compute_allowed_tuples(4, c_prod)
    assert (2, 3, 2) in tuples
    for t in tuples:
        assert cage_satisfied(c_prod, list(t))

