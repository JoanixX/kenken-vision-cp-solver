#!/usr/bin/env python3
"""Script de Benchmarking para el Solver KenKen CP-SAT.

Evalúa y compara sistemáticamente:
  - Variante A (Aritmética Intensional)
  - Variante A + Redundante (Conservación de sumas lineales)
  - Variante B (Extensional / Restricciones de Tabla con AddAllowedAssignments)
  - Variante B + Redundante

Uso:
  python scripts/benchmark.py --sizes 3 4 5 6 7 8 9 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Asegurar que el paquete kenken sea importable desde la raíz del repo
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from kenken.benchmark import run_benchmark, plot_benchmark, VARIANTS

DEFAULT_CSV = ROOT_DIR / "results" / "benchmarks" / "cp_benchmark.csv"
DEFAULT_FIG = ROOT_DIR / "results" / "figs" / "cp_benchmark.png"


def main():
    parser = argparse.ArgumentParser(description="Benchmarking del Solver KenKen CP-SAT")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[3, 4, 5, 6, 7, 8, 9],
        help="Órdenes de grilla n a evaluar",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Número de instancias aleatorias por tamaño",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(DEFAULT_CSV),
        help="Ruta del archivo CSV de salida",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default=str(DEFAULT_FIG),
        help="Ruta de la figura PNG generada",
    )
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_fig = Path(args.out_fig)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    print(f"Iniciando benchmark de CP para n={args.sizes} ({args.repeats} repeticiones)...")
    df = run_benchmark(sizes=tuple(args.sizes), repeats=args.repeats)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Resultados guardados en: {out_csv}")

    # Guardar copia de retrocompatibilidad si es la ruta por defecto
    compat_csv = ROOT_DIR / "results" / "cp_benchmark.csv"
    if out_csv == DEFAULT_CSV:
        df.to_csv(compat_csv, index=False)

    plot_benchmark(df, out_fig)
    print(f"[OK] Figura generada en: {out_fig}")

    summary = df.groupby(["n", "variant"])[["wall_time_ms", "branches", "conflicts"]].mean().reset_index()
    print("\nResumen de Métricas (Medias):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
