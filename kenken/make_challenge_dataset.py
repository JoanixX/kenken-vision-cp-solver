"""Generador de Dataset de Estrés y Desafío (Challenge Stress Dataset).

Construye un conjunto de tableros KenKen donde la lectura directa visual Top-1
NO es 100% consistente, provocando que el solver deba activar Inferencia Conjunta
(Variante C / MAP) para corregir lecturas ambiguas, o presentando casos extremos
de degradación física y confusión de caracteres para entrenamientos futuros.

Uso:
    python -m kenken.make_challenge_dataset --count 50 --out dataset/challenge_stress
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from .generator import generate
from .pipeline import solve_image
from .render import render_sample


def generate_challenge_dataset(
    out_dir: str | Path = "dataset/challenge_stress",
    target_count: int = 50,
    seed: int = 42,
    prefix: str = "stress",
    max_attempts: int = 300,
    verbose: bool = True,
) -> list[dict]:
    """Genera un dataset donde ningún tablero tenga lectura directa Top-1 100% consistente."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    collected = []
    attempts = 0
    t0 = time.time()

    if verbose:
        print(f"Iniciando generación de {target_count} muestras de estrés en '{out_dir}'...")

    while len(collected) < target_count and attempts < max_attempts:
        attempts += 1
        n = rng.randint(3, 8)
        inst, sol = generate(n, seed=rng.randrange(2**31))

        # Renderizar con degradación física y fuentes desafiantes
        level = "hard" if rng.random() < 0.85 else "normal"
        img, gt = render_sample(inst, sol, rng, photo=True, level=level)

        try:
            res = solve_image(img, method="auto")
        except Exception:
            continue

        # Criterio fundamental: la lectura directa Top-1 NO debe ser consistente
        # (es decir, fallback_used es True, o el estado no es solved)
        is_challenge = False
        challenge_type = ""

        if res.solved and res.fallback_used:
            is_challenge = True
            challenge_type = "map_correction"  # Resuelto mediante Inferencia Conjunta (MAP)
        elif not res.solved and res.status in ("infeasible", "invalid_instance"):
            is_challenge = True
            challenge_type = "severe_ambiguity"  # Caso de estrés extremo para futuro fine-tuning

        if not is_challenge:
            # Descartar si el Top-1 fue 100% consistente sin necesidad de corrección
            continue

        idx = len(collected)
        stem = out_path / f"{prefix}_{idx:05d}"
        img_file = stem.with_suffix(".jpg")
        json_file = stem.with_suffix(".json")

        gt["challenge_meta"] = {
            "type": challenge_type,
            "fallback_used": res.fallback_used,
            "solver_status": res.status,
            "attempts_before": attempts,
            "corrections_needed": len(res.solve_result.chosen_candidates) if res.solve_result and res.solve_result.chosen_candidates else 0,
        }
        gt["image"]["file"] = img_file.name

        # Guardar imagen y ground truth
        cv2.imwrite(str(img_file), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        collected.append({
            "id": idx,
            "file": img_file.name,
            "n": n,
            "challenge_type": challenge_type,
            "fallback_used": res.fallback_used,
            "status": res.status,
            "corrections": gt["challenge_meta"]["corrections_needed"],
        })

        if verbose:
            print(
                f"  [{len(collected):2d}/{target_count}] {img_file.name} (n={n}) -> "
                f"Tipo: {challenge_type:18s} | Fallback: {str(res.fallback_used):5s} | Status: {res.status}",
                flush=True,
            )

    # Guardar manifiesto de todo el dataset
    manifest = {
        "dataset_name": "Challenge Stress Dataset",
        "total_samples": len(collected),
        "generation_time_s": round(time.time() - t0, 2),
        "seed": seed,
        "criteria": "Direct Top-1 reading is NOT 100% consistent (fallback_used==True or status!=solved)",
        "summary": {
            "map_correction_count": sum(1 for c in collected if c["challenge_type"] == "map_correction"),
            "severe_ambiguity_count": sum(1 for c in collected if c["challenge_type"] == "severe_ambiguity"),
            "size_distribution": {n: sum(1 for c in collected if c["n"] == n) for n in range(3, 9)},
        },
        "samples": collected,
    }
    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n[OK] Generadas {len(collected)} muestras en {time.time() - t0:.1f}s.")
        print(f"Manifiesto guardado en: {out_path / 'manifest.json'}")

    return collected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera el dataset de estrés con fallos en lectura Top-1.")
    parser.add_argument("--count", type=int, default=50, help="Número de muestras de desafío")
    parser.add_argument("--out", default="dataset/challenge_stress", help="Directorio de salida")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="stress")
    args = parser.parse_args()

    generate_challenge_dataset(
        out_dir=args.out,
        target_count=args.count,
        seed=args.seed,
        prefix=args.prefix,
    )
