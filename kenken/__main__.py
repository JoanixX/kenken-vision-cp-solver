"""Resuelve una instancia JSON desde la terminal:  python -m kenken examples/4x4_a.json"""

import sys

from .instance import Instance
from .model import format_grid, has_unique_solution, solve

inst = Instance.load(sys.argv[1]).validate()
print(inst, "\n")
res = solve(inst)
print(f"estado={res.status}  tiempo={res.wall_time:.4f}s  "
      f"ramas={res.branches}  conflictos={res.conflicts}")
if res.solved:
    print(format_grid(res.grid))
    print("solución única:", has_unique_solution(inst))
