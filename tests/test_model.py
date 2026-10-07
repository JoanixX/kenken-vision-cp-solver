"""Pruebas del hito 1: instancia JSON, validación, modelo CP y generador.

Ejecutar desde TP/:   python -m pytest -q
"""

from pathlib import Path

import pytest

from kenken.generator import generate
from kenken.instance import Cage, Instance, InstanceError
from kenken.model import check_solution, find_solutions, has_unique_solution, solve

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
