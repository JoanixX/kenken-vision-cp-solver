"""Punto de entrada de Streamlit para despliegues desde la raíz del repositorio.

Permite que plataformas como Streamlit Community Cloud (share.streamlit.io)
o Hugging Face detecten y ejecuten la aplicación automáticamente desde la raíz.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Asegurar que la raíz y prototipo_interactivo estén en sys.path
ROOT_DIR = Path(__file__).resolve().parent
PROT_DIR = ROOT_DIR / "prototipo_interactivo"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PROT_DIR) not in sys.path:
    sys.path.insert(0, str(PROT_DIR))

# Importar y ejecutar la interfaz de Streamlit
from prototipo_interactivo.app import run_streamlit_app, resolver_kenken, resolver_kenken_detallado, build_app

if __name__ == "__main__":
    run_streamlit_app()
