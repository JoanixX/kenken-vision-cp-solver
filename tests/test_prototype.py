from pathlib import Path
import cv2
import numpy as np
import pytest

from prototipo_interactivo.app import build_app, resolver_kenken


def test_build_app_structure():
    """Verifica que la aplicación de Gradio se construya correctamente."""
    demo = build_app()
    assert demo is not None
    assert hasattr(demo, "blocks")


def test_resolver_kenken_with_valid_image():
    """Verifica que resolver_kenken procese un array de imagen y devuelva solución y métricas."""
    sample_path = Path("prototipo_interactivo/sample_images/kenken_4x4.png")
    assert sample_path.exists(), "La imagen de muestra 4x4 debe existir"

    img_bgr = cv2.imread(str(sample_path))
    assert img_bgr is not None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    sol_rgb, info_md = resolver_kenken(img_rgb, method="auto", render_mode="composite")

    assert sol_rgb is not None
    assert isinstance(sol_rgb, np.ndarray)
    assert sol_rgb.ndim == 3 and sol_rgb.shape[2] == 3
    assert "Resuelto Exitosamente" in info_md
    assert "OPTIMAL" in info_md


def test_resolver_kenken_with_none_image():
    """Verifica que resolver_kenken maneje entradas nulas de forma segura."""
    sol, info = resolver_kenken(None, method="auto", render_mode="composite")
    assert sol is None
    assert "captura una fotografía" in info


def test_resolver_kenken_multiple_adjustments_warning():
    """Verifica que el prototipo muestre aviso de verificación cuando hay múltiples ajustes."""
    from unittest.mock import MagicMock, patch
    from kenken.instance import Cage, Instance
    from kenken.model import SolveResult
    from kenken.pipeline import PipelineResult

    mock_inst = Instance(4, [Cage([[0, 0]], 1, "=")])
    mock_solve = MagicMock(spec=SolveResult)
    mock_solve.solved = True
    mock_solve.status = "OPTIMAL"
    mock_solve.grid = [[1, 2], [2, 1]]
    mock_solve.wall_time = 0.05
    mock_solve.branches = 10
    mock_solve.conflicts = 2
    mock_solve.chosen_candidates = {}

    mock_res = MagicMock(spec=PipelineResult)
    mock_res.solved = True
    mock_res.grid = [[1, 2], [2, 1]]
    mock_res.status = "OPTIMAL"
    mock_res.fallback_used = True
    mock_res.divergent = True
    mock_res.num_changed = 4
    mock_res.fidelity = 60.0
    mock_res.confidence_level = "MODERATE"
    mock_res.instance = mock_inst
    mock_res.solve_result = mock_solve
    mock_res.render_solution.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("prototipo_interactivo.app.solve_image", return_value=mock_res):
        sol_rgb, info_md = resolver_kenken(fake_img, method="auto", render_mode="composite")
        assert sol_rgb is not None
        assert "OPTIMAL" in info_md
        assert "Aviso de Verificación" in info_md
        assert "4 correcciones" in info_md


def test_resolver_kenken_infeasible():
    """Verifica que el prototipo reporte infactibilidad cuando no hay solución."""
    from unittest.mock import MagicMock, patch
    from kenken.pipeline import PipelineResult

    mock_res = MagicMock(spec=PipelineResult)
    mock_res.solved = False
    mock_res.grid = None
    mock_res.status = "INFEASIBLE"

    fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("prototipo_interactivo.app.solve_image", return_value=mock_res):
        sol_rgb, info_md = resolver_kenken(fake_img, method="auto", render_mode="composite")
        assert sol_rgb is None
        assert "INFEASIBLE" in info_md
        assert "No se pudo encontrar una solución válida" in info_md


def test_all_showcase_cases_exist():
    """Verifica que las imágenes de muestra para los casos del prototipo existan y sean válidas."""
    expected_cases = [
        "caso1_lectura_directa_4x4.png",
        "caso2_rescate_map_4x4.jpg",
        "caso3_perspectiva_rotacion_5x5.jpg",
        "caso4_alta_dificultad_6x6.jpg",
        "caso5_gran_escala_9x9.png",
        "caso6_rescate_avanzado_6x6.jpg",
        "caso7_aviso_ajustes_multiples_6x6.jpg",
        "caso8_infactible_contradictorio.jpg",
    ]
    sample_dir = Path("prototipo_interactivo/sample_images")
    for case_name in expected_cases:
        p = sample_dir / case_name
        assert p.exists(), f"El archivo de muestra {case_name} debe existir"
        img = cv2.imread(str(p))
        assert img is not None, f"El archivo {case_name} debe ser una imagen válida"



