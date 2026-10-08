"""Resuelve una instancia JSON o una imagen de KenKen desde la terminal.

Ejemplos de uso:
  python -m kenken examples/4x4_a.json
  python -m kenken examples/4x4_a.json --output-img results/4x4_sol.png
  python -m kenken dataset/synthetic/syn_00000.png
  python -m kenken dataset/synthetic/syn_00000.png --output-img results/syn_sol.png --render-mode composite
"""

import argparse
import sys
from pathlib import Path

from .instance import Instance
from .model import format_grid, has_unique_solution, solve
from .visualize import draw_solution_clean


def main():
    parser = argparse.ArgumentParser(
        description="KenKen Solver: Visión Computacional + Constraint Programming (OR-Tools CP-SAT)"
    )
    parser.add_argument("path", type=str, help="Ruta al archivo JSON o a la imagen (.jpg, .png)")
    parser.add_argument(
        "-o",
        "--output-img",
        type=str,
        default=None,
        help="Ruta donde guardar la imagen con la solución del KenKen resuelto",
    )
    parser.add_argument(
        "--render-mode",
        choices=["composite", "clean", "rectified", "original"],
        default="composite",
        help="Modo de renderizado: composite (lado a lado), clean (grilla limpia), original (proyectada), rectified",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "arithmetic", "table", "joint"],
        default="auto",
        help="Estrategia del solver CP (default: auto)",
    )
    parser.add_argument(
        "--redundant",
        action="store_true",
        help="Activar restricciones redundantes de suma triangular",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta a los pesos del modelo OCR (default: models/ocr_cnn.pt)",
    )

    args = parser.parse_args()
    arg_path = Path(args.path)

    if not arg_path.exists():
        print(f"Error: El archivo no existe: {arg_path}", file=sys.stderr)
        sys.exit(1)

    if arg_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        from .pipeline import solve_image

        print(f"Resolviendo imagen: {arg_path}...")
        res_pipe = solve_image(
            arg_path,
            method=args.method,
            redundant=args.redundant,
            cnn_model=args.model,
            output_image=args.output_img,
            render_mode=args.render_mode,
        )
        print(res_pipe.instance, "\n")
        print(
            f"estado={res_pipe.status}  fallback_conjunto={res_pipe.fallback_used}  "
            f"tiempo={res_pipe.solve_result.wall_time:.4f}s  "
            f"ramas={res_pipe.solve_result.branches}  conflictos={res_pipe.solve_result.conflicts}"
        )
        if res_pipe.solved:
            print("\nGrilla solución:")
            print(format_grid(res_pipe.grid))
            if args.output_img:
                print(f"\n[OK] Imagen de la solución guardada en: {args.output_img}")
    else:
        inst = Instance.load(arg_path).validate()
        print(inst, "\n")
        res = solve(inst, redundant=args.redundant)
        print(
            f"estado={res.status}  tiempo={res.wall_time:.4f}s  "
            f"ramas={res.branches}  conflictos={res.conflicts}"
        )
        if res.solved:
            print("\nGrilla solución:")
            print(format_grid(res.grid))
            print("solución única:", has_unique_solution(inst))
            if args.output_img:
                draw_solution_clean(inst, res.grid, out_path=args.output_img)
                print(f"\n[OK] Imagen de la solución guardada en: {args.output_img}")


if __name__ == "__main__":
    main()
