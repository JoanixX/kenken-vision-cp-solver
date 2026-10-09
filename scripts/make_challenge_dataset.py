#!/usr/bin/env python3
"""Script de Generación de Dataset de Estrés y Desafío (Challenge Stress Dataset).

Construye un conjunto de tableros KenKen donde la lectura directa visual Top-1
NO es 100% consistente, provocando que el solver deba activar Inferencia Conjunta
(Variante C / MAP) para corregir lecturas ambiguas.

Uso:
  python scripts/make_challenge_dataset.py --count 50 --out dataset/challenge_stress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kenken.make_challenge_dataset import generate_challenge_dataset


def main():
    parser = argparse.ArgumentParser(description="Generador de dataset de estrés y fallos controlados para KenKen")
    parser.add_argument("--count", type=int, default=50, help="Número objetivo de tableros desafiantes")
    parser.add_argument("--out", type=str, default="dataset/challenge_stress", help="Directorio de salida")
    parser.add_argument("--seed", type=int, default=42, help="Semilla generadora")
    parser.add_argument("--prefix", type=str, default="stress", help="Prefijo de los archivos generados")
    parser.add_argument("--max-attempts", type=int, default=300, help="Límite máximo de intentos de muestreo")
    args = parser.parse_args()

    generate_challenge_dataset(
        out_dir=args.out,
        target_count=args.count,
        seed=args.seed,
        prefix=args.prefix,
        max_attempts=args.max_attempts,
        verbose=True,
    )


if __name__ == "__main__":
    main()
