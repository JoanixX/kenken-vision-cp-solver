"""Módulo de benchmarking y perfilado experimental para Constraint Programming (CP).

Evalúa y compara sistemáticamente:
  - Variante A (Aritmética Intensional)
  - Variante A + Redundante (Conservación de sumas lineales)
  - Variante B (Extensional / Restricciones de Tabla con AddAllowedAssignments)
  - Variante B + Redundante

Métricas registradas:
  - Tiempo de resolución (Wall time en ms)
  - Ramas exploradas en el árbol de búsqueda (Branches)
  - Conflictos resueltos por propagación CDCL/LCG (Conflicts)
  - Verificación matemática de unicidad y exactitud

Uso:
  python -m kenken.benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .generator import generate
from .model import check_solution, solve

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figs"

VARIANTS = [
    ("A_aritmética", "arithmetic", False),
    ("A_redundante", "arithmetic", True),
    ("B_tabla", "table", False),
    ("B_redundante", "table", True),
]


def run_benchmark(
    sizes: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9),
    repeats: int = 3,
    seed_base: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Ejecuta el benchmark sistemático sobre múltiples instancias generadas."""
    records = []

    for n in sizes:
        if verbose:
            print(f"--> Evaluando tableros n={n} ({repeats} instancias)...")
        for rep in range(repeats):
            seed = seed_base + rep * 100 + n
            inst, true_sol = generate(n, seed=seed)
            num_cages = len(inst.cages)

            for label, var_name, redundant in VARIANTS:
                res = solve(inst, variant=var_name, redundant=redundant)
                is_valid = res.solved and check_solution(inst, res.grid) and (res.grid == true_sol)

                records.append({
                    "n": n,
                    "repeat": rep,
                    "seed": seed,
                    "cages": num_cages,
                    "variant": label,
                    "method": var_name,
                    "redundant": redundant,
                    "status": res.status,
                    "wall_time_ms": res.wall_time * 1000.0,
                    "branches": res.branches,
                    "conflicts": res.conflicts,
                    "is_valid": is_valid,
                })

    df = pd.DataFrame(records)
    return df


def plot_benchmark(df: pd.DataFrame, out_path: Path = FIGS_DIR / "cp_benchmark.png"):
    """Genera una figura comparativa de alta calidad (3 paneles) para el informe técnico IEEE."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    ax_time, ax_search, ax_bar = axes

    # Estilos y colores por variante
    styles = {
        "A_aritmética": ("#1f77b4", "o-", "A: Aritmética"),
        "A_redundante": ("#aec7e8", "s--", "A + Redundante"),
        "B_tabla": ("#2ca02c", "^-", "B: Tabla (GAC)"),
        "B_redundante": ("#98df8a", "d--", "B + Redundante"),
    }

    # ------------------------------------------------------------- Panel 1: Tiempo vs n
    for var_label, (color, fmt, display_name) in styles.items():
        sub = df[df["variant"] == var_label]
        grouped = sub.groupby("n")["wall_time_ms"].agg(["mean", "std"]).reset_index()
        ax_time.plot(grouped["n"], grouped["mean"], fmt, color=color, label=display_name, lw=2, ms=6)
        if grouped["std"].notna().any():
            ax_time.fill_between(
                grouped["n"],
                grouped["mean"] - grouped["std"].fillna(0),
                grouped["mean"] + grouped["std"].fillna(0),
                color=color,
                alpha=0.15,
            )

    ax_time.set_title("Tiempo de Resolución vs. Orden $n$", fontsize=12, fontweight="bold")
    ax_time.set_xlabel("Tamaño del Tablero ($n$)", fontsize=11)
    ax_time.set_ylabel("Wall Time medio (ms)", fontsize=11)
    ax_time.grid(True, linestyle="--", alpha=0.4)
    ax_time.legend(frameon=True, fontsize=9)

    # ------------------------------------------------ Panel 2: Espacio Teórico vs Ramas
    ns = sorted(df["n"].unique())
    theoretical_log10 = [float(n * n * np.log10(n)) for n in ns]  # log10(n^(n^2))
    
    # Ramas medias exploradas
    mean_branches_a = df[df["variant"] == "A_aritmética"].groupby("n")["branches"].mean().reindex(ns)
    mean_branches_b = df[df["variant"] == "B_tabla"].groupby("n")["branches"].mean().reindex(ns)

    ax_search.plot(ns, theoretical_log10, "r*--", lw=1.8, label="Espacio bruto $\\log_{10}(n^{n^2})$")
    ax_search_twin = ax_search.twinx()
    ax_search_twin.plot(ns, mean_branches_a, "o-", color="#1f77b4", lw=2, label="Ramas Aritmética")
    ax_search_twin.plot(ns, mean_branches_b, "^-", color="#2ca02c", lw=2, label="Ramas Tabla")

    ax_search.set_title("Espacio Teórico vs. Ramas Exploradas", fontsize=12, fontweight="bold")
    ax_search.set_xlabel("Tamaño del Tablero ($n$)", fontsize=11)
    ax_search.set_ylabel("$\\log_{10}(\\text{Espacio de Estados})$", color="darkred", fontsize=11)
    ax_search_twin.set_ylabel("Ramas Exploradas por CP-SAT", color="#1f77b4", fontsize=11)
    ax_search.grid(True, linestyle="--", alpha=0.4)

    # Combinar leyendas de ejes gemelos
    lines1, labels1 = ax_search.get_legend_handles_labels()
    lines2, labels2 = ax_search_twin.get_legend_handles_labels()
    ax_search.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8.5)

    # ---------------------------------------------------- Panel 3: Comparativa de Tiempos
    piv = df.pivot_table(index="n", columns="variant", values="wall_time_ms", aggfunc="mean")
    x_indices = np.arange(len(ns))
    width = 0.2

    for idx, (var_label, (color, _, display_name)) in enumerate(styles.items()):
        vals = piv[var_label].values if var_label in piv else [0] * len(ns)
        ax_bar.bar(x_indices + (idx - 1.5) * width, vals, width, label=display_name, color=color, alpha=0.85)

    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels([f"{n}x{n}" for n in ns])
    ax_bar.set_title("Comparativa Empírica de Variantes", fontsize=12, fontweight="bold")
    ax_bar.set_xlabel("Dimensiones del Puzzle", fontsize=11)
    ax_bar.set_ylabel("Wall Time medio (ms)", fontsize=11)
    ax_bar.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_bar.legend(frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmarking del Solver KenKen CP-SAT")
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8, 9], help="Órdenes de grilla n a evaluar")
    parser.add_argument("--repeats", type=int, default=3, help="Número de instancias aleatorias por tamaño")
    parser.add_argument("--out-csv", type=str, default=str(RESULTS_DIR / "cp_benchmark.csv"), help="Ruta del archivo CSV")
    parser.add_argument("--out-fig", type=str, default=str(FIGS_DIR / "cp_benchmark.png"), help="Ruta de la figura generada")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Iniciando benchmark de CP para n={args.sizes} ({args.repeats} repeticiones)...")
    df = run_benchmark(sizes=tuple(args.sizes), repeats=args.repeats)
    df.to_csv(args.out_csv, index=False)
    print(f"Resultados guardados en: {args.out_csv}")

    plot_benchmark(df, Path(args.out_fig))
    print(f"Figura generada en: {args.out_fig}")

    # Mostrar resumen
    summary = df.groupby(["n", "variant"])[["wall_time_ms", "branches", "conflicts"]].mean().reset_index()
    print("\nResumen de Métricas (Medias):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
