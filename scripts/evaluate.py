#!/usr/bin/env python3
"""Script de Evaluación Cuantitativa del Pipeline KenKen.

Permite evaluar:
  1. Detección de estructura (tablero, tamaño n y partición de jaulas)
  2. OCR de etiquetas aritméticas (precisión top-1 y top-k, matriz de confusión)
  3. Rendimiento integral end-to-end (imagen a solución verificada)

Uso:
  python scripts/evaluate.py dataset/synthetic --structure
  python scripts/evaluate.py dataset/synthetic --ocr
  python scripts/evaluate.py dataset/synthetic --e2e
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kenken.evaluate import (
    evaluate_structure,
    evaluate_ocr,
    evaluate_end2end,
    summarize,
    summarize_ocr,
    summarize_end2end,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluación cuantitativa de visión, OCR y solver end-to-end")
    parser.add_argument("dataset", type=str, help="Ruta al directorio del dataset (ej: dataset/synthetic)")
    parser.add_argument("--structure", action="store_true", help="Evaluar detección de estructura geométrica")
    parser.add_argument("--ocr", action="store_true", help="Evaluar reconocimiento óptico de caracteres")
    parser.add_argument("--e2e", action="store_true", help="Evaluar pipeline de punta a punta")
    parser.add_argument("--out-csv", type=str, default=None, help="Ruta para exportar métricas en CSV")
    parser.add_argument("--no-labels", dest="use_labels", action="store_false", default=True,
                        help="No usar etiquetas numéricas para ayudar a resolver empates en jaulas")
    parser.add_argument("--no-alt", dest="alternatives", action="store_false", default=True,
                        help="No extraer candidatos alternativos en OCR")

    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"Error: El dataset no existe en {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Si no se seleccionó ninguna bandera explícita, evaluar estructura por defecto
    if not (args.structure or args.ocr or args.e2e):
        args.structure = True

    if args.e2e:
        print(f"--> Evaluando pipeline End-to-End en {dataset_path}...")
        results = evaluate_end2end(dataset_path, out_csv=args.out_csv)
        print(summarize_end2end(results))
    elif args.ocr:
        print(f"--> Evaluando OCR en {dataset_path}...")
        rows, conf = evaluate_ocr(dataset_path, out_csv=args.out_csv, alternatives=args.alternatives)
        print(summarize_ocr(rows, conf))
    else:
        print(f"--> Evaluando Estructura Geométrica en {dataset_path}...")
        rows = evaluate_structure(dataset_path, out_csv=args.out_csv, use_labels=args.use_labels)
        print(summarize(rows))


if __name__ == "__main__":
    main()
