# Documentación del Módulo de Constraint Programming (CP)

> **Módulo:** `kenken.model`  
> **Proyecto:** KenKen Solver — Integración Visión Computacional + Constraint Programming  
> **Curso:** CC58 - Tópicos en Ciencia de la Computación  

---

## 1. Visión General del Módulo CP

El módulo de Constraint Programming es el núcleo lógico del sistema. Su objetivo es recibir la representación estructurada del tablero (extraída en la fase de visión o cargada desde un archivo JSON) y encontrar la única asignación consistente de números en la grilla que satisfaga simultáneamente:
1. Las restricciones globales del **Cuadrado Latino** (filas y columnas con dígitos del $1$ al $n$ sin repetición).
2. Las restricciones aritméticas locales de cada **jaula** (*cage*).

Para cumplir con la rúbrica de evaluación y permitir un estudio experimental comparativo, el módulo se estructura en tres variantes de modelado:
* **Variante A (Aritmética Intensional):** Descompone las jaulas en restricciones elementales (sumas lineales, productos encadenados, diferencias absolutas y divisiones reificadas).
* **Variante B (Extensional / Restricciones de Tabla):** Modela cada jaula mediante la restricción global `AddAllowedAssignments` utilizando catálogos de tuplas factibles calculadas a priori.
* **Variante C (Inferencia Conjunta MAP):** Optimización combinatoria que selecciona entre hipótesis alternativas de OCR mediante variables indicadoras reificadas con `OnlyEnforceIf`.

---

## 2. Tarea 1: Generador de Tuplas Factibles (`compute_allowed_tuples`)

### 2.1. Motivación y Rol en Constraint Programming
En los problemas de satisfacción de restricciones (CSP), una restricción sobre un conjunto de variables $V = (x_1, \dots, x_k)$ puede definirse de forma **intensional** (a través de una fórmula lógica/aritmética) o de forma **extensional** (mediante una tabla que enumera explícitamente las tuplas válidas permitidas).

La función [`compute_allowed_tuples(n, cage)`](../kenken/model.py) construye el catálogo de tuplas permitidas $\mathcal{T}_{\text{cage}} \subset \{1, \dots, n\}^k$ para una jaula con $k$ celdas. Esta representación es indispensable para:
1. Alimentar la restricción global `AddAllowedAssignments` (Variante B).
2. Permitir que el solver aplique **Consistencia de Arco Generalizada (GAC)** sobre toda la jaula antes de iniciar la búsqueda arborizada, reduciendo el tamaño de los dominios sin necesidad de introducir variables intermedias auxiliares.

---

### 2.2. Fundamento Matemático y Topológico

Sea una jaula $C = \langle \text{cells}, T, \text{op} \rangle$, donde $\text{cells} = \{(r_0, c_0), (r_1, c_1), \dots, (r_{k-1}, c_{k-1})\}$, con objetivo $T \in \mathbb{N}$ y operador $\text{op} \in \{=, +, -, \times, \div\}$.

Una tupla $\mathbf{v} = (v_0, v_1, \dots, v_{k-1}) \in \{1, \dots, n\}^k$ es **factible** si y solo si cumple dos condiciones simultáneas:

#### Condición 1: Respeto al Cuadrado Latino (Filtrado Topológico)
Aunque las celdas pertenezcan a la misma jaula, están sujetas a las restricciones globales de fila y columna del tablero completo. Por lo tanto:
$$\forall i, j \in \{0, \dots, k-1\}, i \neq j: \quad (r_i = r_j \lor c_i = c_j) \implies v_i \neq v_j$$

* **Celdas Colineales:** Si varias celdas comparten la misma fila o la misma columna, sus valores deben ser estrictamente diferentes.
* **Celdas No Alineadas (Diagonales o Formas en 'L'):** Si dos celdas no comparten fila ni columna, **sí pueden tener el mismo valor**. Por ejemplo, en una jaula de suma $7$ con celdas $(0,0), (0,1), (1,1)$, la asignación $(2, 3, 2)$ es perfectamente válida porque las dos celdas con valor $2$ no interfieren entre sí.

#### Condición 2: Satisfacción del Operador Aritmético
* **Identidad ($=$):** $k = 1 \land v_0 = T$.
* **Resta ($-$):** $k = 2 \land |v_0 - v_1| = T$.
* **División ($\div$):** $k = 2 \land (v_0 = T \cdot v_1 \lor v_1 = T \cdot v_0)$.
* **Suma ($+$):** $\sum_{i=0}^{k-1} v_i = T$.
* **Multiplicación ($\times$):** $\prod_{i=0}^{k-1} v_i = T$.

---

### 2.3. Algoritmo de Filtrado y Poda Anticipada

Para evitar el costo computacional de explorar el producto cartesiano completo $\{1..n\}^k$ (incluso cuando $n \le 9$), la función utiliza un algoritmo de búsqueda por retroceso (*backtracking*) con **poda anticipada de ramas infactibles**:

```
Entrada: n, cage = (cells, target, op)
Salida: Lista de tuplas factibles allowed

1. Precalcular grafo de conflictos internos:
   Para cada celda i de 0 a k-1:
     conflicts[i] = {j < i | cells[i].fila == cells[j].fila o cells[i].col == cells[j].col}

2. Si op == '=':
   Si k == 1 y 1 <= target <= n: retornar [(target,)]
   Sino: retornar []

3. Si op == '-' o op == '/':
   Generar pares (a, b) directamente en O(n), verificando no colisión si comparten fila/columna.

4. Si op == '+':
   Buscar recursivamente (idx, suma_actual):
     Cotas:
       rem = k - idx
       min_posible = suma_actual + rem * 1
       max_posible = suma_actual + rem * n
       Si target < min_posible o target > max_posible: podar rama (return).
     Para val de 1 a n:
       Si val colisiona con conflicts[idx]: omitir.
       Si suma_actual + val + (rem - 1) * 1 > target: romper bucle (break).
       Añadir val y descender a idx + 1.

5. Si op == '*':
   Buscar recursivamente (idx, prod_actual):
     Si target % prod_actual != 0: podar rama (return).
     rem_target = target // prod_actual
     Si rem_target > (n ^ rem): podar rama (return).
     Para val de 1 a n:
       Si rem_target % val != 0: omitir (podar no divisores).
       Si val colisiona con conflicts[idx]: omitir.
       Añadir val y descender a idx + 1.
```

---

### 2.4. Ejemplos de Salida y Casos Críticos

#### Caso A: Jaula Colineal vs. Jaula en 'L' (Suma 7, Grilla 4×4)
* **Jaula Colineal:** `cells = [(0,0), (0,1), (0,2)]` (todas en fila 0).
  * Tuplas resultantes ($6$ tuplas):
    $$(1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1)$$
  * Observe que ninguna tupla contiene números repetidos.
* **Jaula en 'L':** `cells = [(0,0), (0,1), (1,1)]`.
  * Tuplas resultantes ($8$ tuplas):
    $$(1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 3, 2), (2, 4, 1), (3, 1, 3), (4, 1, 2), (4, 2, 1)$$
  * Las tuplas $(2, 3, 2)$ y $(3, 1, 3)$ son permitidas porque las celdas $(0,0)$ y $(1,1)$ no se solapan en fila ni columna.

#### Caso B: Multiplicación con Factores Repetidos
* **Jaula en 'L':** `cells = [(0,0), (0,1), (1,1)]`, `target = 12`, `op = '*'`.
  * Genera $(2, 3, 2)$ puesto que $2 \times 3 \times 2 = 12$, respetando las restricciones de la grilla.

---

### 2.5. Validación y Pruebas Unitarias

La implementación cuenta con pruebas exhaustivas en [`tests/test_model.py`](../tests/test_model.py):
* `test_compute_allowed_tuples_equals`: Valida targets válidos y descartes fuera de rango.
* `test_compute_allowed_tuples_subtraction`: Verifica la conmutatividad y el filtrado en restas.
* `test_compute_allowed_tuples_division`: Verifica divisiones válidas en ambos sentidos.
* `test_compute_allowed_tuples_addition_collinear_vs_l_shape`: Comprueba que las jaulas colineales descarten repeticiones y las no colineales las permitan.
* `test_compute_allowed_tuples_multiplication`: Comprueba factorización y verificación de satisfacción independiente vía `cage_satisfied`.

**Resultado de ejecución de pruebas:**
```bash
python -m pytest tests/test_model.py -q
# 27 passed in 0.85s (100% éxito)
```
