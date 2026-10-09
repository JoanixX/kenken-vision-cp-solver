"""Configuración y fixtures compartidas de Pytest para KenKen Vision-CP Solver."""

import sys
from pathlib import Path
import pytest

# Asegurar que el directorio raíz del repositorio siempre esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Retorna la ruta absoluta a la raíz del repositorio."""
    return ROOT_DIR


@pytest.fixture(scope="session")
def sample_4x4_json(repo_root) -> Path:
    """Ruta a la instancia de prueba 4x4 en formato JSON."""
    return repo_root / "examples" / "4x4_a.json"


@pytest.fixture(scope="session")
def models_dir(repo_root) -> Path:
    """Ruta al directorio de modelos preentrenados."""
    return repo_root / "models"


@pytest.fixture(scope="session")
def results_dir(repo_root) -> Path:
    """Ruta al directorio de resultados."""
    return repo_root / "results"
