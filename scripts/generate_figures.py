#!/usr/bin/env python3
"""Script de Generación de Figuras para Informe Técnico y Documentación.

Genera:
  - Curvas de entrenamiento de la CNN
  - Matrices de confusión de OCR (sintético y hard)
  - Curvas de ajuste fino

Uso:
  python scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kenken.cnn import MODEL_PATH
from kenken.figures import training_curves, confusion_matrix, FIGS

FIGS_DIR = ROOT_DIR / "results" / "figs"
RESULTS_DIR = ROOT_DIR / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print("--> Generando curvas de entrenamiento base...")
    training_curves(out=FIGS_DIR / "cnn_training.png")

    # Matrices de confusión si los CSVs están presentes
    csv_candidates = [
        ("ocr_synthetic.confusion.csv", "ocr_synthetic_confusion.png", "OCR Sintético (modelo base)"),
        ("ocr_synthetic_hard.confusion.csv", "ocr_synthetic_hard_confusion.png", "OCR Sintético Hard (modelo base)"),
        ("ocr_finetuned_synthetic.confusion.csv", "ocr_finetuned_synthetic_confusion.png", "OCR Sintético (fine-tuned)"),
        ("ocr_finetuned_hard.confusion.csv", "ocr_finetuned_hard_confusion.png", "OCR Sintético Hard (fine-tuned)"),
    ]

    for csv_name, out_img_name, title in csv_candidates:
        # Buscar en results/ y results/benchmarks/
        p = RESULTS_DIR / csv_name
        if not p.exists():
            p = BENCHMARKS_DIR / csv_name
        if p.exists():
            print(f"--> Generando matriz de confusión: {out_img_name}...")
            confusion_matrix(p, FIGS_DIR / out_img_name, title)

    print(f"[OK] Figuras actualizadas en {FIGS_DIR}")


if __name__ == "__main__":
    main()
