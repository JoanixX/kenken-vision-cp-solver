"""Generador aleatorio de instancias KenKen con solución única.

Algoritmo:
  1. Cuadrado latino aleatorio (la solución oculta).
  2. Partición aleatoria del tablero en jaulas conexas (crecimiento aleatorio).
  3. A cada jaula se le asigna una operación compatible con sus valores y se
     calcula el objetivo a partir de la solución oculta.
  4. Se descarta la instancia si el solver encuentra más de una solución.

Sirve para entrenar/evaluar la visión (render sintético) y para el benchmark
del solver en función de n.
"""

from __future__ import annotations

import random

from .instance import Cage, Instance
from .model import Grid, has_unique_solution

# Probabilidad del tamaño de cada jaula (tamaños típicos de KenKen publicados).
SIZE_WEIGHTS = {1: 0.10, 2: 0.50, 3: 0.30, 4: 0.10}


def random_latin_square(n: int, rng: random.Random) -> Grid:
    """Cuadrado latino cíclico con filas, columnas y símbolos permutados al azar.

    No es uniforme sobre todos los cuadrados latinos, pero es suficiente
    para generar puzzles variados.
    """
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    rows, cols, syms = list(range(n)), list(range(n)), list(range(1, n + 1))
    rng.shuffle(rows), rng.shuffle(cols), rng.shuffle(syms)
    return [[syms[base[r][c]] for c in cols] for r in rows]


def random_partition(n: int, rng: random.Random,
                     size_weights: dict[int, float] = SIZE_WEIGHTS) -> list[list[tuple[int, int]]]:
    """Divide el tablero en jaulas conexas haciéndolas crecer desde celdas libres."""
    free = {(i, j) for i in range(n) for j in range(n)}
    sizes, weights = list(size_weights), list(size_weights.values())
    cages = []
    while free:
        start = min(free)  # recorrer en orden de lectura deja menos huecos aislados
        target_size = rng.choices(sizes, weights)[0]
        cage = [start]
        free.remove(start)
        while len(cage) < target_size:
            frontier = [nb for (i, j) in cage
                        for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                        if nb in free]
            if not frontier:
                break
            nxt = rng.choice(frontier)
            free.remove(nxt)
            cage.append(nxt)
        cages.append(sorted(cage))
    return cages


def choose_operation(values: list[int], rng: random.Random, max_product: int = 2000) -> tuple[str, int]:
    """Elige una operación válida para los valores de la jaula y devuelve (op, objetivo)."""
    if len(values) == 1:
        return "=", values[0]

    prod = 1
    for v in values:
        prod *= v
    options = [("+", sum(values))]
    if prod <= max_product:
        options.append(("*", prod))

    if len(values) == 2:
        a, b = max(values), min(values)
        options.append(("-", a - b))
        if a % b == 0:
            # Las divisiones exactas son poco frecuentes: se priorizan cuando aparecen.
            options += [("/", a // b)] * 2
    return rng.choice(options)


def generate(n: int, seed: int | None = None, unique: bool = True,
             max_tries: int = 200) -> tuple[Instance, Grid]:
    """Genera una instancia válida de tamaño n y su solución.

    Si unique=True, reintenta hasta que la instancia tenga solución única.
    """
    rng = random.Random(seed)
    for _ in range(max_tries):
        solution = random_latin_square(n, rng)
        cages = []
        for cells in random_partition(n, rng):
            op, target = choose_operation([solution[i][j] for i, j in cells], rng)
            cages.append(Cage(cells, target, op))
        inst = Instance(n, cages).validate()
        if not unique or has_unique_solution(inst):
            return inst, solution
    raise RuntimeError(f"no se logró una instancia única de n={n} en {max_tries} intentos")
