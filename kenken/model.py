"""Modelo de Constraint Programming (OR-Tools CP-SAT) para KenKen.

Formulación (variante A):

  Variables   x[i][j] ∈ {1..n}           valor de la celda (i, j)
  Globales    AllDifferent(fila i)       para cada fila     (cuadrado latino)
              AllDifferent(columna j)    para cada columna
  Jaulas      '='  x == T
              '+'  Σ x == T                              (lineal)
              '*'  Π x == T                              (AddMultiplicationEquality encadenado)
              '-'  |a - b| == T                          (AddAbsEquality)
              '/'  b_dir → a == T·c ;  ¬b_dir → c == T·a (reificada con OnlyEnforceIf)
  Opcional    Σ fila == n(n+1)/2 y Σ columna == n(n+1)/2 (redundante, implicada por AllDifferent)
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .instance import Cage, Instance

Grid = list[list[int]]


# ============================================================ construcción
def build_model(inst: Instance, redundant: bool = False):
    """Crea el modelo CP-SAT a partir de la instancia.

    Devuelve (model, x) donde x[i][j] es la variable de la celda (i, j).
    """
    n = inst.n
    model = cp_model.CpModel()

    # Variables y dominios: igual que el cuadrado latino de la semana 3.
    x = [[model.NewIntVar(1, n, f"x{i}{j}") for j in range(n)] for i in range(n)]

    # Restricciones globales: cada fila y cada columna es una permutación de 1..n.
    for i in range(n):
        model.AddAllDifferent(x[i])
    for j in range(n):
        model.AddAllDifferent([x[i][j] for i in range(n)])

    if redundant:
        total = n * (n + 1) // 2
        for i in range(n):
            model.Add(sum(x[i]) == total)
        for j in range(n):
            model.Add(sum(x[i][j] for i in range(n)) == total)

    for k, cage in enumerate(inst.cages):
        add_cage_constraint(model, x, cage, name=f"c{k}")

    return model, x


def add_cage_constraint(model: cp_model.CpModel, x, cage: Cage, name: str = "c"):
    """Agrega la restricción aritmética de una jaula.

    Se separa en una función porque la variante de inferencia conjunta la
    reutiliza activándola solo si se elige esa lectura (OnlyEnforceIf).
    """
    v = [x[i][j] for i, j in cage.cells]
    t, op = cage.target, cage.op

    if op == "=":
        model.Add(v[0] == t)

    elif op == "+":
        model.Add(sum(v) == t)

    elif op == "*":
        # CP-SAT multiplica de a dos: p1 = v0·v1, p2 = p1·v2, ...  y el último == T.
        # Las variables intermedias tienen dominio [1, n^k].
        n = len(x)
        acc = v[0]
        for idx, var in enumerate(v[1:], start=2):
            if idx == len(v):
                model.AddMultiplicationEquality(t, [acc, var])
            else:
                p = model.NewIntVar(1, n ** idx, f"{name}_p{idx}")
                model.AddMultiplicationEquality(p, [acc, var])
                acc = p

    elif op == "-":
        model.AddAbsEquality(t, v[0] - v[1])

    elif op == "/":
        # No sabemos cuál de las dos celdas es el numerador: lo decide un booleano.
        # big=True  → v0 == T·v1   (v0 es el mayor)
        # big=False → v1 == T·v0   (v1 es el mayor)
        big = model.NewBoolVar(f"{name}_dir")
        model.Add(v[0] == t * v[1]).OnlyEnforceIf(big)
        model.Add(v[1] == t * v[0]).OnlyEnforceIf(big.Not())

    else:  # pragma: no cover - Cage ya valida op
        raise ValueError(op)


# ============================================================ resolución
@dataclass
class SolveResult:
    status: str                 # OPTIMAL / FEASIBLE / INFEASIBLE / MODEL_INVALID / UNKNOWN
    grid: Grid | None
    wall_time: float
    branches: int
    conflicts: int

    @property
    def solved(self) -> bool:
        return self.grid is not None


def solve(inst: Instance, redundant: bool = False, time_limit: float = 30.0,
          workers: int = 8, seed: int = 0) -> SolveResult:
    """Resuelve la instancia y devuelve la grilla y las estadísticas del solver."""
    model, x = build_model(inst, redundant=redundant)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(model)

    grid = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        grid = [[solver.Value(v) for v in row] for row in x]
    return SolveResult(solver.StatusName(status), grid, solver.WallTime(),
                       solver.NumBranches(), solver.NumConflicts())


class _SolutionCounter(cp_model.CpSolverSolutionCallback):
    """Cuenta soluciones y detiene la búsqueda al llegar a `limit`."""

    def __init__(self, x, limit: int):
        super().__init__()
        self._x, self._limit = x, limit
        self.solutions: list[Grid] = []

    def on_solution_callback(self):
        self.solutions.append([[self.Value(v) for v in row] for row in self._x])
        if len(self.solutions) >= self._limit:
            self.StopSearch()


def find_solutions(inst: Instance, limit: int = 2, time_limit: float = 30.0) -> list[Grid]:
    """Enumera hasta `limit` soluciones distintas (con limit=2 basta para saber si es única)."""
    model, x = build_model(inst)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.enumerate_all_solutions = True  # requiere 1 worker
    counter = _SolutionCounter(x, limit)
    solver.Solve(model, counter)
    return counter.solutions


def has_unique_solution(inst: Instance, time_limit: float = 30.0) -> bool:
    return len(find_solutions(inst, limit=2, time_limit=time_limit)) == 1


# ============================================================ verificación
def check_solution(inst: Instance, grid: Grid) -> bool:
    """Verificación independiente del solver (Python puro)."""
    n = inst.n
    full = set(range(1, n + 1))
    if any(set(row) != full for row in grid):
        return False
    if any({grid[i][j] for i in range(n)} != full for j in range(n)):
        return False
    return all(cage_satisfied(c, [grid[i][j] for i, j in c.cells]) for c in inst.cages)


def cage_satisfied(cage: Cage, values: list[int]) -> bool:
    t, op = cage.target, cage.op
    if op == "=":
        return values[0] == t
    if op == "+":
        return sum(values) == t
    if op == "*":
        prod = 1
        for v in values:
            prod *= v
        return prod == t
    a, b = values
    if op == "-":
        return abs(a - b) == t
    if op == "/":
        return a == t * b or b == t * a
    return False


def format_grid(grid: Grid) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)

