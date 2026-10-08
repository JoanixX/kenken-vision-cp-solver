# Plan de Implementación: Módulo de Constraint Programming (CP)

> **Documento de Trabajo:** Guía paso a paso para la implementación, verificación y evaluación del módulo de Constraint Programming para el proyecto KenKen Solver.  
> **Objetivo de Rúbrica:** 7 puntos de Modelado CP (Formulación formal, Restricciones Globales, Restricciones Reificadas) + Soporte para Integración y Sección Experimental del Informe Técnico.

---

## 1. Resumen de Variantes de Modelado

| Variante | Tipo | Descripción | Restricciones Globales / Reificadas |
|---|---|---|---|
| **Variante A** | Intensional (Aritmética) | Formulación canónica descomponiendo jaulas en operadores elementales. | `AllDifferent` (filas/cols), suma lineal, producto encadenado, división reificada con `OnlyEnforceIf(b)`. |
| **Variante A + Redundante** | Intensional + Implicadas | Añade sumas de conservación global por fila y columna ($\sum = n(n+1)/2$). | Analizar si reduce ramas o agrega sobrecosto de propagación. |
| **Variante B** | Extensional (Tabla) | Modela cada jaula mediante el catálogo de tuplas válidas calculadas a priori. | `AllDifferent` + `AddAllowedAssignments` (propagación GAC sobre jaulas completas). |
| **Variante C** | Inferencia Conjunta (MAP) | Optimización combinatoria que selecciona entre hipótesis top-$k$ del OCR. | `ExactlyOne` por jaula + `OnlyEnforceIf(r[c,k])` para cada hipótesis + función objetivo $\max \sum \text{score}$. |

---

## 2. Checklist de Tareas Paso a Paso

- [x] **Tarea 1: Generador de Tuplas Factibles para Jaulas**
  - [x] Implementar `compute_allowed_tuples(n, cage)` en `kenken/model.py`.
  - [x] Validar descarte de tuplas con números repetidos en celdas alineadas (misma fila o columna).
  - [x] Pruebas unitarias para operadores `=`, `+`, `-`, `*`, `/`.

- [x] **Tarea 2: Implementación de la Variante B (`build_model_table`)**
  - [x] Implementar `build_model_table(inst, redundant=False)` en `kenken/model.py`.
  - [x] Integrar `AddAllowedAssignments` para cada jaula.
  - [x] Pruebas de equivalencia: verificar que Variante B entrega la misma solución que Variante A en todos los ejemplos de `examples/`.

- [x] **Tarea 3: Implementación de la Variante C (`solve_joint`) e Inferencia Conjunta**
  - [x] Implementar `add_reified_cage_constraint(model, x, cage, r_var, name)` en `kenken/model.py`.
  - [x] Implementar `build_model_joint(inst)` y `solve_joint(inst, ...)` en `kenken/model.py`.
  - [x] Soportar función objetivo ponderada por $\log P$ de la CNN.
  - [x] Pruebas con ruido simulado: forzar un error en el target u operador top-1 y verificar que el solver recupera la solución correcta usando el top-$k$.

- [x] **Tarea 4: Integración en el Pipeline End-to-End con Fallback Automático**
  - [x] Actualizar `kenken/pipeline.py` para definir `solve_image(path, method="auto", ...)`.
  - [x] Estrategia "auto": intentar primero resolver con la lectura top-1 (Variante A); si resulta `INFEASIBLE`, ejecutar automáticamente Variante C (`solve_joint`).
  - [x] Validar que el flujo completo no requiera intervención manual.

- [x] **Tarea 5: Suite de Pruebas Unitarias Exhaustivas de CP**
  - [x] Crear/actualizar `tests/test_model.py` con pruebas para:
    - [x] Variante A vs Variante B (consistencia de resultados).
    - [x] Detección de instancias infactibles.
    - [x] Conteo de unicidad de soluciones (`has_unique_solution`).
    - [x] Robustez de la Variante C ante lecturas de OCR degradadas.

- [x] **Tarea 6: Módulo de Benchmarking y Perfilado Experimental (`kenken/benchmark.py`)**
  - [x] Crear script para evaluar rendimiento variando:
    - [x] Tamaño de tablero $n \in \{3, 4, 5, 6, 7, 8, 9\}$.
    - [x] Variante de modelado: A, A+redundante, B, B+redundante.
  - [x] Registrar métricas: Wall time (ms), `NumBranches()`, `NumConflicts()`, número de jaulas.
  - [x] Exportar resultados a `results/cp_benchmark.csv`.
  - [x] Generar gráficos comparativos en `results/figs/cp_benchmark.png` (tiempo vs $n$, ramas vs $n$, comparativa).

- [ ] **Tarea 7: Redacción del Modelo Formal y Análisis de Complejidad para el Informe LaTeX**
  - [ ] Redactar especificación formal $\langle X, D, C \rangle$ con LaTeX.
  - [ ] Explicar la formulación de restricciones globales y reificación en la sección técnica.
  - [ ] Documentar análisis de complejidad teórica ($n^{n^2}$) vs empírica (espacio explorado por CP-SAT con Lazy Clause Generation).

---

## 3. Especificación Técnica Detallada

### 3.1. Tarea 1: Generador de Tuplas Factibles (`compute_allowed_tuples`)

**Ubicación:** `kenken/model.py`  
**Firma:**
```python
def compute_allowed_tuples(n: int, cage: Cage) -> list[tuple[int, ...]]:
    """Calcula todas las tuplas (v_1, ..., v_k) permitidas para la jaula.
    
    Condiciones:
    1. Cada v_i in {1..n}.
    2. Si dos celdas comparten fila o columna, sus valores deben ser distintos.
    3. El conjunto de valores satisface (target, op).
    """
```

**Lógica de filtrado:**
- Generar tuplas con `itertools.product(range(1, n + 1), repeat=k)`.
- Restricción de posición: para cada par de índices $i < j$, si `cells[i][0] == cells[j][0]` o `cells[i][1] == cells[j][1]`, descartar si `v[i] == v[j]`.
- Restricción aritmética: evaluar según el operador:
  - `=`: $v_0 == T$
  - `+`: $\sum v == T$
  - `*`: $\prod v == T$
  - `-`: $|v_0 - v_1| == T$
  - `/`: $v_0 == T \cdot v_1 \lor v_1 == T \cdot v_0$

---

### 3.2. Tarea 2: Variante B (Restricciones de Tabla)

**Ubicación:** `kenken/model.py`  
**Firma:**
```python
def build_model_table(inst: Instance, redundant: bool = False):
    """Variante B: Modela las jaulas con AddAllowedAssignments."""
```

**Características:**
- Variables $x[i][j] \in \{1..n\}$.
- Restricciones globales `AddAllDifferent` en filas y columnas.
- Para cada jaula:
  ```python
  scope = [x[i][j] for i, j in cage.cells]
  tuples = compute_allowed_tuples(inst.n, cage)
  model.AddAllowedAssignments(scope, tuples)
  ```
- Devuelve `(model, x)`.

---

### 3.3. Tarea 3: Variante C (Inferencia Conjunta MAP)

**Ubicación:** `kenken/model.py`  
**Firma:**
```python
def build_model_joint(inst: Instance, scale: int = 1000):
    """Variante C: Modela inferencia conjunta sobre hipótesis candidates[c]."""
```

**Mapeo de Reificación:**
Para cada lectura candidata $k$ de la jaula $c$ con variable booleana $r_{c,k}$:
- `+`: `model.Add(sum(vars) == target).OnlyEnforceIf(r)`
- `=`: `model.Add(vars[0] == target).OnlyEnforceIf(r)`
- `-`: 
  ```python
  d = model.NewIntVar(-inst.n, inst.n, f"diff_{name}")
  model.Add(d == vars[0] - vars[1]).OnlyEnforceIf(r)
  model.AddAbsEquality(target, d).OnlyEnforceIf(r)
  ```
- `/`:
  ```python
  b_dir = model.NewBoolVar(f"dir_{name}")
  model.Add(vars[0] == target * vars[1]).OnlyEnforceIf([r, b_dir])
  model.Add(vars[1] == target * vars[0]).OnlyEnforceIf([r, b_dir.Not()])
  ```
- `*`: intermediarios $p_i$ condicionados con `OnlyEnforceIf(r)`.

**Función Objetivo:**
$$\max \sum_{c} \sum_{k} \lfloor \text{scale} \cdot \log P_{c,k} \rfloor \cdot r_{c,k}$$

---

### 3.4. Tarea 4: Pipeline y Fallback Automático

**Ubicación:** `kenken/pipeline.py`  
**Flujo de `solve_image`:**
```python
def solve_image(img_or_path, use_joint_fallback: bool = True):
    st = extract_structure(img)
    inst = read_instance(st)
    res = solve(inst)
    if not res.solved and use_joint_fallback and inst.candidates:
        res = solve_joint(inst)
    return res, st, inst
```

---

### 3.5. Tarea 6: Benchmarking (`kenken/benchmark.py`)

**Parámetros del experimento:**
- Instancias: Generar con `generate(n, seed=s)` para $n \in \{3, 4, 5, 6, 7, 8\}$ con 5 semillas por tamaño.
- Métricas a registrar por corrida:
  1. `instance_id`, `n`, `num_cages`
  2. `variant`: `A`, `A_redundant`, `B`
  3. `status`: `OPTIMAL` / `FEASIBLE`
  4. `wall_time_sec`: tiempo real de resolución
  5. `branches`: ramas exploradas en el árbol de búsqueda
  6. `conflicts`: número de conflictos resueltos
  7. `num_variables`, `num_constraints`
- Salidas:
  - `results/cp_benchmark.csv`
  - `results/figs/cp_time_vs_n.png`
  - `results/figs/cp_branches_vs_n.png`

---

## 4. Criterios de Aceptación

1. **Exactitud:** Variante A, Variante A+Redundante y Variante B encuentran soluciones idénticas para todas las instancias de prueba en `examples/`.
2. **Robustez de Inferencia Conjunta:** Al inyectar un error de lectura en una jaula (p. ej. operador o target alterado pero con la opción correcta en el top-3), `solve_joint` encuentra la configuración correcta sin intervención manual.
3. **Reproducibilidad:** El benchmark corre con un comando (`python -m kenken.benchmark`) y genera los gráficos listos para ser incluidos en el informe LaTeX.
