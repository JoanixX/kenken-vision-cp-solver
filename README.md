# TB1 — KenKen Solver (Visión Computacional + Constraint Programming)

Ver `PLAN.md` para el diseño completo.

## Instalación

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/Mac/Colab
```

## Uso (hito 1: modelo CP)

```bash
python -m kenken examples/4x4_a.json     # resuelve una instancia JSON
python -m pytest -q                      # pruebas
```

Desde Python:

```python
from kenken import Instance, solve
from kenken.generator import generate

inst = Instance.load("examples/5x5_a.json").validate()
res = solve(inst)            # res.grid, res.status, res.wall_time, res.branches, res.conflicts

inst, sol = generate(7, seed=1)   # puzzle aleatorio 7x7 con solución única
```

## Estructura actual

| Archivo | Contenido |
|---|---|
| `kenken/instance.py` | Formato JSON de la instancia (`Instance`, `Cage`) y validación estructural |
| `kenken/model.py` | Modelo CP-SAT (variante A), conteo de soluciones, verificador independiente |
| `kenken/generator.py` | Generador de puzzles aleatorios con solución única |
| `examples/` | 3 instancias escritas a mano (3x3, 4x4, 5x5) + 1 generada (6x6, seed 6) |
| `tests/` | Pruebas con pytest |
