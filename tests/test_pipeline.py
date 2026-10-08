"""Pruebas del pipeline end-to-end: solve_image (imagen -> estructura -> OCR -> CP)."""

import random
from unittest.mock import patch

import pytest

from kenken.generator import generate
from kenken.instance import Cage, Instance
from kenken.pipeline import PipelineResult, read_instance, solve_image
from kenken.render import render_sample


def test_solve_image_synthetic_end_to_end():
    inst, sol = generate(4, seed=42)
    img, gt = render_sample(inst, sol, random.Random(42), photo=False)
    res = solve_image(img, method="auto")

    assert isinstance(res, PipelineResult)
    assert res.solved
    assert res.status == "OPTIMAL"
    assert res.grid == sol
    assert res.structure.n == 4
    assert len(res.structure.cages) == len(inst.cages)


@pytest.mark.parametrize("method", ["arithmetic", "table", "joint"])
def test_solve_image_methods(method):
    inst, sol = generate(4, seed=10)
    img, gt = render_sample(inst, sol, random.Random(10), photo=False)
    res = solve_image(img, method=method)

    assert res.solved
    assert res.grid == sol


def test_solve_image_auto_triggers_fallback():
    """Si la lectura top-1 es infactible, el método 'auto' debe activar

    automáticamente el fallback a la inferencia conjunta (Variante C).
    """
    inst, sol = generate(4, seed=20)
    img, gt = render_sample(inst, sol, random.Random(20), photo=False)

    real_read_instance = read_instance

    def mock_read_instance(st, **kwargs):
        real_inst = real_read_instance(st, **kwargs)
        # Forzar que el top-1 de la jaula 0 tenga un target infactible (999)
        # y colocar la lectura real en el top-2
        true_cage = real_inst.cages[0]
        corrupted_cages = list(real_inst.cages)
        corrupted_cages[0] = Cage(true_cage.cells, 999, true_cage.op)

        candidates = dict(real_inst.candidates)
        candidates[0] = [
            {"target": 999, "op": true_cage.op, "logp": -0.01},
            {"target": true_cage.target, "op": true_cage.op, "logp": -0.60},
        ]
        return Instance(real_inst.n, corrupted_cages, candidates)

    with patch("kenken.pipeline.read_instance", side_effect=mock_read_instance):
        res = solve_image(img, method="auto")
        assert res.solved
        assert res.fallback_used
        assert res.grid == sol


def test_solve_image_file_not_found():
    with pytest.raises(FileNotFoundError, match="No se pudo cargar la imagen"):
        solve_image("archivo_inexistente_123456.png")


def test_solve_image_invalid_method():
    inst, sol = generate(3, seed=1)
    img, _ = render_sample(inst, sol, random.Random(1), photo=False)
    with pytest.raises(ValueError, match="Método desconocido"):
        solve_image(img, method="metodo_invalido")
