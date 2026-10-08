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
