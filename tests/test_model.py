"""Pruebas del hito 1: instancia JSON, validación, modelo CP y generador.

Ejecutar desde TP/:   python -m pytest -q
"""

from pathlib import Path

import pytest

from kenken.generator import generate
from kenken.instance import Cage, Instance, InstanceError
from kenken.model import (
    build_model,
    build_model_table,
    cage_satisfied,
    check_solution,
    compute_allowed_tuples,
    find_solutions,
    has_unique_solution,
    solve,
    solve_joint,
    solve_table,
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


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_solve_uniquely_table_variant(path):
    inst = Instance.load(path).validate()
    res = solve(inst, variant="table")
    assert res.status == "OPTIMAL"
    assert check_solution(inst, res.grid)
    assert has_unique_solution(inst, variant="table")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_table_variant_matches_arithmetic_variant(path):
    inst = Instance.load(path).validate()
    res_arith = solve(inst, variant="arithmetic")
    res_table = solve(inst, variant="table")
    assert res_arith.grid == res_table.grid
    assert res_table.status == "OPTIMAL"


@pytest.mark.parametrize("redundant", [False, True])
def test_table_variant_redundant_constraint(redundant):
    inst = Instance.load(EXAMPLES[1])
    res = solve_table(inst, redundant=redundant)
    assert res.status == "OPTIMAL"
    assert check_solution(inst, res.grid)


def test_table_variant_infeasible():
    inst = Instance(3, [Cage([[0, 0], [0, 1], [0, 2]], 7, "+"),
                        Cage([[1, 0], [1, 1], [1, 2], [2, 2]], 9, "+"),
                        Cage([[2, 0], [2, 1]], 3, "+")])
    assert solve(inst, variant="table").status == "INFEASIBLE"


def test_solve_invalid_variant():
    inst = Instance.load(EXAMPLES[0])
    with pytest.raises(ValueError, match="Variante desconocida"):
        solve(inst, variant="nonexistent")


def test_solve_joint_with_clean_candidates():
    inst = Instance.load(EXAMPLES[0])
    cands = {
        c_idx: [{"target": cage.target, "op": cage.op, "logp": 0.0}]
        for c_idx, cage in enumerate(inst.cages)
    }
    inst_with_cands = Instance(inst.n, inst.cages, cands)
    res = solve_joint(inst_with_cands)
    assert res.status == "OPTIMAL"
    assert check_solution(inst, res.grid)
    assert res.chosen_candidates is not None
    for c_idx in range(len(inst.cages)):
        assert res.chosen_candidates[c_idx]["target"] == inst.cages[c_idx].target


def test_solve_joint_recovers_from_corrupted_top1_target():
    inst = Instance.load(EXAMPLES[1])  # 4x4 instance
    true_sol = solve(inst).grid

    true_cage = inst.cages[0]
    corrupted_cages = list(inst.cages)
    corrupted_cages[0] = Cage(true_cage.cells, 999, true_cage.op)

    cands = {
        0: [
            {"target": 999, "op": true_cage.op, "logp": -0.01},  # corrupto
            {"target": true_cage.target, "op": true_cage.op, "logp": -0.80},  # correcto
        ]
    }
    for c_idx in range(1, len(inst.cages)):
        c = inst.cages[c_idx]
        cands[c_idx] = [{"target": c.target, "op": c.op, "logp": 0.0}]

    noisy_inst = Instance(inst.n, corrupted_cages, cands)

    res_direct = solve(noisy_inst, variant="arithmetic")
    assert res_direct.status == "INFEASIBLE"

    res_joint = solve_joint(noisy_inst)
    assert res_joint.status == "OPTIMAL"
    assert res_joint.grid == true_sol
    assert res_joint.chosen_candidates[0]["target"] == true_cage.target


def test_solve_joint_recovers_from_corrupted_top1_operator():
    inst = Instance.load(EXAMPLES[1])
    true_sol = solve(inst).grid

    prod_idx = [i for i, c in enumerate(inst.cages) if c.op == "*"][0]
    true_cage = inst.cages[prod_idx]

    corrupted_cages = list(inst.cages)
    corrupted_cages[prod_idx] = Cage(true_cage.cells, true_cage.target, "+")

    cands = {
        prod_idx: [
            {"target": true_cage.target, "op": "+", "logp": -0.05},
            {"target": true_cage.target, "op": "*", "logp": -0.50},
        ]
    }
    for c_idx in range(len(inst.cages)):
        if c_idx != prod_idx:
            c = inst.cages[c_idx]
            cands[c_idx] = [{"target": c.target, "op": c.op, "logp": 0.0}]

    noisy_inst = Instance(inst.n, corrupted_cages, cands)
    res_joint = solve(noisy_inst, variant="joint")
    assert res_joint.status == "OPTIMAL"
    assert res_joint.grid == true_sol
    assert res_joint.chosen_candidates[prod_idx]["op"] == "*"


def test_solve_joint_when_no_candidates_provided():
    inst = Instance.load(EXAMPLES[0])
    inst_no_cands = Instance(inst.n, inst.cages, {})
    res = solve_joint(inst_no_cands)
    assert res.status == "OPTIMAL"
    assert check_solution(inst, res.grid)


def test_solve_joint_infeasible_when_all_candidates_impossible():
    inst = Instance.load(EXAMPLES[0])
    cands = {0: [{"target": 999, "op": "+", "logp": 0.0}]}
    inst_bad = Instance(inst.n, inst.cages, cands)
    assert solve_joint(inst_bad).status == "INFEASIBLE"


def test_solve_joint_tracks_num_changed():
    inst = Instance.load(EXAMPLES[1])
    true_sol = solve(inst).grid

    true_cage = inst.cages[0]
    corrupted_cages = list(inst.cages)
    corrupted_cages[0] = Cage(true_cage.cells, 999, true_cage.op)

    cands = {
        0: [
            {"target": 999, "op": true_cage.op, "logp": -0.01},  # top-1 (erróneo)
            {"target": true_cage.target, "op": true_cage.op, "logp": -0.50},  # top-2 (correcto)
        ]
    }
    for c_idx in range(1, len(inst.cages)):
        c = inst.cages[c_idx]
        cands[c_idx] = [{"target": c.target, "op": c.op, "logp": 0.0}]

    noisy_inst = Instance(inst.n, corrupted_cages, cands)
    res_joint = solve_joint(noisy_inst)
    assert res_joint.status == "OPTIMAL"
    assert res_joint.grid == true_sol
    assert res_joint.num_changed == 1


def test_solve_joint_zero_changes_when_clean():
    inst = Instance.load(EXAMPLES[1])
    cands = {i: [{"target": c.target, "op": c.op, "logp": 0.0}] for i, c in enumerate(inst.cages)}
    clean_inst = Instance(inst.n, inst.cages, cands)
    res_joint = solve_joint(clean_inst)
    assert res_joint.status == "OPTIMAL"
    assert res_joint.num_changed == 0




