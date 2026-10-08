"""Resuelve una instancia JSON o una imagen desde la terminal:
  python -m kenken examples/4x4_a.json
  python -m kenken puzzle.png
"""

import sys
from pathlib import Path

from .instance import Instance
from .model import format_grid, has_unique_solution, solve

arg_path = Path(sys.argv[1])
if arg_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
    from .pipeline import solve_image

    print(f"Resolviendo imagen: {arg_path}...")
    res_pipe = solve_image(arg_path)
    print(res_pipe.instance, "\n")
    print(f"estado={res_pipe.status}  fallback_conjunto={res_pipe.fallback_used}  "
          f"tiempo={res_pipe.solve_result.wall_time:.4f}s  "
          f"ramas={res_pipe.solve_result.branches}  conflictos={res_pipe.solve_result.conflicts}")
    if res_pipe.solved:
        print(format_grid(res_pipe.grid))
else:
    inst = Instance.load(arg_path).validate()
    print(inst, "\n")
    res = solve(inst)
    print(f"estado={res.status}  tiempo={res.wall_time:.4f}s  "
          f"ramas={res.branches}  conflictos={res.conflicts}")
    if res.solved:
        print(format_grid(res.grid))
        print("solución única:", has_unique_solution(inst))
