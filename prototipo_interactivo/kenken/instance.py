"""Representación intermedia de una instancia KenKen (puente visión -> CP).

Una instancia es un tablero n x n dividido en jaulas. Cada jaula tiene:
  - cells:  lista de celdas (fila, columna), indexadas desde 0
  - target: número objetivo
  - op:     operación, una de  '+', '-', '*', '/', '='

Formato JSON (el mismo que produce la fase de visión):

    {
      "n": 4,
      "cages": [
        {"cells": [[0,0],[1,0]], "target": 3, "op": "/"},
        {"cells": [[3,3]],       "target": 1, "op": "="}
      ]
    }

Opcionalmente puede traer "candidates" (lecturas alternativas de la CNN), que
se usan en la variante de inferencia conjunta del modelo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

OPS = ("+", "-", "*", "/", "=")

# Símbolos que puede devolver el OCR o escribir una persona -> operación canónica.
OP_ALIASES = {
    "+": "+",
    "-": "-", "−": "-", "–": "-",
    "*": "*", "x": "*", "X": "*", "×": "*", "·": "*",
    "/": "/", "÷": "/", ":": "/",
    "=": "=", "": "=", None: "=",
}

Cell = tuple[int, int]


class InstanceError(ValueError):
    """La instancia no cumple las reglas de KenKen. `errors` lista cada problema."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Instancia inválida:\n  - " + "\n  - ".join(errors))


@dataclass
class Cage:
    cells: list[Cell]
    target: int
    op: str

    def __post_init__(self):
        self.cells = [tuple(c) for c in self.cells]
        self.target = int(self.target)
        if self.op not in OP_ALIASES:
            raise InstanceError([f"operación desconocida: {self.op!r}"])
        self.op = OP_ALIASES[self.op]

    @property
    def anchor(self) -> Cell:
        """Celda donde va impresa la etiqueta: fila mínima y luego columna mínima."""
        return min(self.cells)

    @property
    def label(self) -> str:
        return str(self.target) if self.op == "=" else f"{self.target}{self.op}"

    def to_dict(self) -> dict:
        return {"cells": [list(c) for c in self.cells], "target": self.target, "op": self.op}


@dataclass
class Instance:
    n: int
    cages: list[Cage]
    candidates: dict[int, list[dict]] = field(default_factory=dict)

    # ---------------------------------------------------------------- E/S JSON
    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        cages = [Cage(c["cells"], c["target"], c.get("op", "=")) for c in d["cages"]]
        cands = {int(k): v for k, v in d.get("candidates", {}).items()}
        return cls(int(d["n"]), cages, cands)

    def to_dict(self) -> dict:
        d = {"n": self.n, "cages": [c.to_dict() for c in self.cages]}
        if self.candidates:
            d["candidates"] = {str(k): v for k, v in self.candidates.items()}
        return d

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    # -------------------------------------------------------------- utilidades
    def cage_of(self) -> dict[Cell, int]:
        """Mapa celda -> índice de su jaula."""
        return {cell: k for k, cage in enumerate(self.cages) for cell in cage.cells}

    def __str__(self) -> str:
        """Dibujo ASCII: cada celda muestra la letra de su jaula."""
        owner = self.cage_of()
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        rows = [" ".join(letters[owner[(i, j)] % len(letters)] for j in range(self.n))
                for i in range(self.n)]
        legend = [f"{letters[k % len(letters)]}: {c.label}" for k, c in enumerate(self.cages)]
        return "\n".join(rows) + "\n" + ", ".join(legend)

    # -------------------------------------------------------------- validación
    def validate(self) -> "Instance":
        """Comprueba la estructura de la instancia. Lanza InstanceError si falla.

        No garantiza que el puzzle tenga solución (eso lo decide el solver);
        solo descarta lecturas imposibles, que suelen ser errores de visión.
        """
        errors: list[str] = []
        n = self.n
        if not 1 <= n <= 9:
            errors.append(f"n={n} fuera de rango (1..9)")

        # 1) Cada celda del tablero pertenece a exactamente una jaula.
        seen: dict[Cell, int] = {}
        for k, cage in enumerate(self.cages):
            if not cage.cells:
                errors.append(f"jaula {k} vacía")
            for cell in cage.cells:
                i, j = cell
                if not (0 <= i < n and 0 <= j < n):
                    errors.append(f"jaula {k}: celda {cell} fuera del tablero")
                elif cell in seen:
                    errors.append(f"celda {cell} está en las jaulas {seen[cell]} y {k}")
                else:
                    seen[cell] = k
        missing = [(i, j) for i in range(n) for j in range(n) if (i, j) not in seen]
        if missing:
            errors.append(f"celdas sin jaula: {missing}")

        for k, cage in enumerate(self.cages):
            if cage.cells:
                errors += [f"jaula {k}: {e}" for e in _cage_errors(cage, n)]

        if errors:
            raise InstanceError(errors)
        return self


def _is_connected(cells: list[Cell]) -> bool:
    """BFS con vecindad-4: ¿todas las celdas de la jaula están unidas?"""
    cells_set = set(cells)
    stack, visited = [cells[0]], {cells[0]}
    while stack:
        i, j = stack.pop()
        for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if nb in cells_set and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(cells_set)


def _cage_errors(cage: Cage, n: int) -> list[str]:
    """Reglas locales de una jaula: conexidad, aridad de la operación y rango del objetivo."""
    errs = []
    size, t, op = len(cage.cells), cage.target, cage.op

    if not _is_connected(cage.cells):
        errs.append("celdas no conexas")
    if op == "=" and size != 1:
        errs.append("'=' requiere exactamente 1 celda")
    if op != "=" and size == 1:
        errs.append(f"jaula de 1 celda con operación {op!r}")
    if op in "-/" and size != 2:
        errs.append(f"{op!r} requiere exactamente 2 celdas (tiene {size})")

    # Cotas simples del objetivo (condiciones necesarias, no suficientes).
    if op == "=" and not 1 <= t <= n:
        errs.append(f"objetivo {t} fuera de 1..{n}")
    elif op == "+" and not size + 1 <= t <= n * size:
        errs.append(f"suma {t} inalcanzable con {size} celdas")
    elif op == "-" and not 1 <= t <= n - 1:
        errs.append(f"resta {t} fuera de 1..{n - 1}")
    elif op == "/" and not 2 <= t <= n:
        errs.append(f"división {t} fuera de 2..{n}")
    elif op == "*":
        if not 2 <= t <= n ** size:
            errs.append(f"producto {t} fuera de rango")
        elif _largest_prime_factor(t) > n:
            errs.append(f"producto {t} tiene un factor primo > {n}")
    return errs


def _largest_prime_factor(m: int) -> int:
    largest, p = 1, 2
    while p * p <= m:
        while m % p == 0:
            largest, m = p, m // p
        p += 1
    return max(largest, m) if m > 1 else largest
