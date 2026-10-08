from pathlib import Path
import cv2
import numpy as np
import pytest

from kenken.instance import Instance
from kenken.model import SolveResult, solve
from kenken.pipeline import PipelineResult, Structure, solve_image
from kenken.visualize import (
    draw_solution_clean,
    overlay_solution_on_original,
    overlay_solution_on_rectified,
    render_solution_composite,
)


@pytest.fixture
def sample_4x4_instance():
    return Instance.load("examples/4x4_a.json")


def test_draw_solution_clean(sample_4x4_instance, tmp_path):
    inst = sample_4x4_instance
    res = solve(inst)
    assert res.solved and res.grid is not None

    out_file = tmp_path / "clean_solved.png"
    img = draw_solution_clean(inst, res.grid, cell_size=80, out_path=out_file)

    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_overlay_solution_on_rectified(sample_4x4_instance, tmp_path):
    inst = sample_4x4_instance
    res = solve(inst)
    n = inst.n
    rect_img = np.full((400, 400, 3), 240, dtype=np.uint8)
    xs = np.linspace(0, 400, n + 1)
    ys = np.linspace(0, 400, n + 1)

    out_file = tmp_path / "rect_overlay.png"
    img = overlay_solution_on_rectified(rect_img, xs, ys, res.grid, out_path=out_file)

    assert isinstance(img, np.ndarray)
    assert img.shape == rect_img.shape
    assert out_file.exists()


def test_overlay_solution_on_original(sample_4x4_instance, tmp_path):
    inst = sample_4x4_instance
    res = solve(inst)
    n = inst.n

    orig_img = np.full((600, 600, 3), 200, dtype=np.uint8)
    corners = np.array([[50, 50], [550, 50], [550, 550], [50, 550]], dtype=np.float32)
    rect = np.full((400, 400, 3), 255, dtype=np.uint8)
    H = cv2.getPerspectiveTransform(corners, np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32))
    xs = np.linspace(0, 400, n + 1)
    ys = np.linspace(0, 400, n + 1)
    st = Structure(corners=corners, H=H, rect=rect, n=n, xs=xs, ys=ys, cages=[c.cells for c in inst.cages])

    out_file = tmp_path / "orig_overlay.png"
    img = overlay_solution_on_original(orig_img, st, res.grid, out_path=out_file)

    assert isinstance(img, np.ndarray)
    assert img.shape == orig_img.shape
    assert out_file.exists()


def test_render_solution_composite(sample_4x4_instance, tmp_path):
    inst = sample_4x4_instance
    res = solve(inst)
    n = inst.n

    orig_img = np.full((500, 500, 3), 220, dtype=np.uint8)
    corners = np.array([[20, 20], [480, 20], [480, 480], [20, 480]], dtype=np.float32)
    rect = np.full((400, 400, 3), 255, dtype=np.uint8)
    H = cv2.getPerspectiveTransform(corners, np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32))
    xs = np.linspace(0, 400, n + 1)
    ys = np.linspace(0, 400, n + 1)
    st = Structure(corners=corners, H=H, rect=rect, n=n, xs=xs, ys=ys, cages=[c.cells for c in inst.cages])

    out_file = tmp_path / "composite.png"
    comp = render_solution_composite(orig_img, inst, res.grid, st=st, out_path=out_file)

    assert isinstance(comp, np.ndarray)
    assert comp.ndim == 3
    assert out_file.exists()


def test_pipeline_result_render_methods(sample_4x4_instance, tmp_path):
    inst = sample_4x4_instance
    res = solve(inst)
    n = inst.n

    orig_img = np.full((400, 400, 3), 255, dtype=np.uint8)
    corners = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
    st = Structure(
        corners=corners,
        H=np.eye(3, dtype=np.float32),
        rect=orig_img,
        n=n,
        xs=np.linspace(0, 400, n + 1),
        ys=np.linspace(0, 400, n + 1),
        cages=[c.cells for c in inst.cages],
    )
    pipe_res = PipelineResult(
        image=orig_img,
        structure=st,
        instance=inst,
        solve_result=res,
    )

    clean_p = tmp_path / "pipe_clean.png"
    pipe_res.save_solution_image(clean_p, mode="clean")
    assert clean_p.exists()

    comp_p = tmp_path / "pipe_comp.png"
    pipe_res.save_solution_image(comp_p, mode="composite")
    assert comp_p.exists()


def test_unsolved_pipeline_result_raises(sample_4x4_instance, tmp_path):
    inst = sample_4x4_instance
    unsolved_res = SolveResult(status="INFEASIBLE", grid=None, wall_time=0.01, branches=0, conflicts=0)
    st = Structure(
        corners=np.zeros((4, 2)),
        H=np.eye(3),
        rect=np.zeros((10, 10, 3), dtype=np.uint8),
        n=inst.n,
        xs=np.zeros(5),
        ys=np.zeros(5),
        cages=[c.cells for c in inst.cages],
    )
    pipe_res = PipelineResult(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        structure=st,
        instance=inst,
        solve_result=unsolved_res,
    )

    with pytest.raises(ValueError, match="No se puede generar la imagen"):
        pipe_res.render_solution(out_path=tmp_path / "fail.png")
