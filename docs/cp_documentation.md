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

---

## 3. Variante B: Restricción Global de Tabla (`AddAllowedAssignments`)

### 3.1. Formulación Matemática de la Variante B

La Variante B modela el problema KenKen como un CSP extensional en las jaulas:

* **Variables de Decisión:**
  $$x_{i,j} \in \{1, \dots, n\} \quad \forall i, j \in \{0, \dots, n-1\}$$

* **Restricciones Globales de Cuadrado Latino:**
  $$\text{AllDifferent}(x_{i,0}, x_{i,1}, \dots, x_{i,n-1}) \quad \forall i \in \{0..n-1\}$$
  $$\text{AllDifferent}(x_{0,j}, x_{1,j}, \dots, x_{n-1,j}) \quad \forall j \in \{0..n-1\}$$

* **Restricción Global Extensional de Jaula:**
  Para cada jaula $c = \langle \text{cells}_c, T_c, \text{op}_c \rangle$, sea $V_c = [x_{i,j} \mid (i,j) \in \text{cells}_c]$ el vector de variables involucradas. Se aplica directamente:
  $$\text{AllowedAssignments}(V_c, \mathcal{T}_c)$$
  donde $\mathcal{T}_c = \text{compute\_allowed\_tuples}(n, c)$.

* **Restricciones Redundantes Opcionales:**
  $$\sum_{j=0}^{n-1} x_{i,j} = \frac{n(n+1)}{2}, \qquad \sum_{i=0}^{n-1} x_{i,j} = \frac{n(n+1)}{2}$$

---

### 3.2. Comparación Teórica: Variante A vs. Variante B

| Característica | Variante A (Aritmética Intensional) | Variante B (Tabla Extensional) |
|---|---|---|
| **Definición de jaula** | Descompuesta en sumas lineales, productos encadenados y booleanos. | Catálogo de tuplas factibles $\mathcal{T}_c$ precalculadas. |
| **Nivel de Consistencia** | Consistencia de Límites (*Bound Consistency*) / Consistencia de Arco local. | **Consistencia de Arco Generalizada (GAC)** sobre la jaula completa. |
| **Variables Auxiliares** | Introduce variables intermedias para producto encadenado ($p_k$) y división ($b_{\text{dir}}$). | **Cero variables auxiliares** en la jaula. |
| **Poda Temprana** | La propagación puede no detectar de inmediato combinaciones cruzadas infactibles. | Poda instantáneamente cualquier valor que no participe en al menos una tupla válida. |
| **Escalabilidad** | Eficiente para jaulas muy grandes si el catálogo de tuplas fuera excesivo. | Óptima para jaulas de tamaño moderado ($k \le 4$), muy comunes en KenKen. |

---

### 3.3. Uso en Código

La función `solve` permite seleccionar la variante mediante el parámetro `variant`:

```python
from kenken import Instance, solve, solve_table

inst = Instance.load("examples/4x4_a.json")

# Variante A (por defecto)
res_a = solve(inst, variant="arithmetic")

# Variante B (restricción global de tabla)
res_b = solve(inst, variant="table")
# o utilizando el alias directo:
res_b = solve_table(inst)

assert res_a.grid == res_b.grid  # Solución 100% idéntica
```

---

### 3.4. Validación y Pruebas Unitarias

Pruebas implementadas en [`tests/test_model.py`](../tests/test_model.py):
* `test_examples_solve_uniquely_table_variant`: Resuelve todas las instancias de prueba en `examples/` (`3x3_a`, `4x4_a`, `5x5_a`, `6x6_gen_seed6`) verificando unicidad y optimalidad con la Variante B.
* `test_table_variant_matches_arithmetic_variant`: Comprueba que la grilla resultante de la Variante B es exactamente igual a la de la Variante A para todas las instancias.
* `test_table_variant_redundant_constraint`: Valida la compatibilidad de restricciones redundantes con tablas.
* `test_table_variant_infeasible`: Comprueba detección inmediata de instancias infactibles (`INFEASIBLE`).
* `test_solve_invalid_variant`: Comprueba el control de errores con variantes desconocidas.

**Resultado de ejecución de pruebas:**
```bash
python -m pytest tests/test_model.py -q
# 39 passed in 0.93s (100% éxito)
```

---

## 4. Variante C: Inferencia Conjunta Neuro-Simbólica (MAP) con Reificación

### 4.1. Motivación y Paradigma Neuro-Simbólico

En un pipeline secuencial clásico (*pipeline ingenuo*):
$$\text{Imagen} \xrightarrow{\text{Visión / CNN}} \text{Lectura Top-1} \xrightarrow{\text{Puente}} \text{Modelo CP} \xrightarrow{\text{Solver}} \text{Solución o INFEASIBLE}$$
Si la CNN confunde un solo carácter (por ejemplo, predice `8-` con 51% de probabilidad cuando el glifo real era `3-` con 49%, o confunde `+` con `*`), el modelo CP tradicional recibe una instancia inconsistente y devuelve **`INFEASIBLE`**, fallando todo el sistema.

La **Variante C** implementa **inferencia conjunta** (*Joint Inference* / *MAP Inference*):
El clasificador de visión computacional no devuelve una sola predicción dura, sino una lista de hipótesis candidatas ordenadas por log-probabilidad:
$$\mathcal{K}_c = \{(T_{c,k}, \text{op}_{c,k}, \log P_{c,k}) \mid k=1, \dots, K\}$$
El solver de CP resuelve simultáneamente dos problemas en uno:
1. **Selección de hipótesis:** Elige la lectura $k$ para cada jaula $c$.
2. **Satisfacción del puzzle:** Asigna números a la grilla que satisfacen el Cuadrado Latino y las jaulas elegidas.

El objetivo es encontrar la lectura **más probable** que **sea consistente con una solución matemática válida**.

---

### 4.2. Formulación Matemática de la Variante C

* **Variables de Decisión Principales:**
  $$x_{i,j} \in \{1, \dots, n\} \quad \forall i, j \in \{0, \dots, n-1\}$$

* **Variables Indicadoras de Lectura (Booleanas):**
  Para cada jaula $c$ y cada candidato $k \in \{0, \dots, K_c - 1\}$:
  $$r_{c, k} \in \{0, 1\}$$
  donde $r_{c, k} = 1$ indica que se adopta la hipótesis $k$ para la jaula $c$.

* **Restricción Global de Exclusividad por Jaula:**
  Cada jaula debe adoptar exactamente una lectura:
  $$\text{ExactlyOne}(\{r_{c, 0}, r_{c, 1}, \dots, r_{c, K_c - 1}\}) \quad \forall c$$

* **Restricciones Aritméticas Reificadas:**
  Las restricciones aritméticas de la jaula se condicionan a $r_{c, k}$ mediante `OnlyEnforceIf`:
  $$r_{c, k} \implies \mathcal{A}(V_c, T_{c,k}, \text{op}_{c,k})$$
  donde:
  - Para suma: $r_{c, k} \implies \sum_{v \in V_c} v = T_{c,k}$
  - Para resta: $r_{c, k} \implies |v_0 - v_1| = T_{c,k}$
  - Para producto: $r_{c, k} \implies \prod_{v \in V_c} v = T_{c,k}$ (encadenado con variables intermedias)
  - Para división: $(r_{c, k} \land b_{\text{dir}}) \implies v_0 = T_{c,k} \cdot v_1$ y $(r_{c, k} \land \neg b_{\text{dir}}) \implies v_1 = T_{c,k} \cdot v_0$
  - Si una hipótesis es matemáticamente imposible en el dominio (ej. $T \le 0$ o división en jaula con 3 casillas), se fuerza $r_{c, k} = 0$.

* **Función Objetivo (MAP / Log-Verosimilitud Máxima):**
  $$\max \sum_{c} \sum_{k} \lfloor \text{scale} \cdot \log P_{c, k} \rfloor \cdot r_{c, k}$$
  donde $\text{scale} = 1000$ convierte los log-probs negativos en coeficientes enteros para CP-SAT.

---

### 4.3. Justificación ante la Rúbrica de Evaluación

La rúbrica del curso asigna 1 punto a:
> *"Uso eficiente de restricciones reificadas (si fuera necesario, sino explicar por qué no)."*

La Variante C proporciona la **máxima justificación técnica posible**:
1. **A nivel aritmético:** Modela la conmutatividad no orientada de la división ($a/b$ o $b/a$) mediante booleanos directos y `OnlyEnforceIf`.
2. **A nivel sistémico / neuro-simbólico:** Emplea reificación para articular el puente entre la incertidumbre del modelo neuronal (CNN) y la rigidez lógica del solver simbólico (CP-SAT), convirtiendo un problema de satisfacción simple en uno de optimización combinatoria que tolera ruido sensorial.

---

### 4.4. Uso en Código

```python
from kenken import Instance, solve, solve_joint

# Instancia con candidatos alternativos en candidates[c]
inst = Instance.load("dataset/noisy_sample.json")

# Resolver directamente con inferencia conjunta
res = solve_joint(inst)
# o equivalentemente:
res = solve(inst, variant="joint")

if res.solved:
    print(f"Estado: {res.status}")
    print("Lecturas elegidas por jaula:")
    for cage_idx, cand in res.chosen_candidates.items():
        print(f"  Jaula {cage_idx}: {cand['target']}{cand['op']} (logp={cand['logp']})")
```

---

### 4.5. Validación y Pruebas Unitarias

Pruebas implementadas en [`tests/test_model.py`](../tests/test_model.py):
* `test_solve_joint_with_clean_candidates`: Verifica que ante lecturas limpias (donde top-1 es la correcta), el modelo selecciona todos los candidatos top-1 y obtiene `OPTIMAL`.
* `test_solve_joint_recovers_from_corrupted_top1_target`: Simula un fallo de OCR donde la lectura top-1 predice un target imposible (`999`) y la lectura top-2 contiene el target verdadero. Verifica que el solver estándar devuelve `INFEASIBLE`, mientras que `solve_joint` se recupera automáticamente eligiendo el top-2 y hallando la grilla correcta.
* `test_solve_joint_recovers_from_corrupted_top1_operator`: Simula una confusión de operador (top-1 predice `+`, top-2 predice `*`). Verifica que `solve_joint` descarta `+`, adopta `*` y resuelve el puzzle.
* `test_solve_joint_when_no_candidates_provided`: Valida el fallback cuando `inst.candidates` está vacío (utiliza las jaulas de `inst.cages`).
* `test_solve_joint_infeasible_when_all_candidates_impossible`: Comprueba que si todas las hipótesis son imposibles, el modelo reporta `INFEASIBLE`.

**Resultado de ejecución de pruebas:**
```bash
python -m pytest tests/test_model.py -q
# 44 passed in 1.20s (100% éxito)
```

---

## 5. Integración End-to-End (`solve_image`) y Fallback Automático

### 5.1. Motivación y Arquitectura del Pipeline

El objetivo de la **Fase 3 del proyecto** es lograr un sistema completamente automatizado (*End-to-End*) donde el usuario proporcione una imagen y el sistema entregue la solución sin requerir ajustes manuales ni reintentos guiados:

$$\text{Imagen (JPG/PNG/Array)} \xrightarrow{\text{solve\_image()}} \text{PipelineResult (Estructura, Instancia, Grilla Solución)}$$

### 5.2. Estrategia de Fallback Automático (`method="auto"`)

La función [`solve_image(img_or_path, method="auto")`](../kenken/pipeline.py) implementa una estrategia jerárquica de resolución:

```mermaid
flowchart TD
    A["Imagen de Entrada (path o np.ndarray)"] --> B["extract_structure(): Homografía, n, Jaulas"]
    B --> C["read_instance(): Glifos + CNN -> Top-k Candidatos"]
    C --> D{"method == 'auto'"}
    D -->|Intento 1| E["solve(inst, variant='arithmetic')"]
    E --> F{¿res.solved?}
    F -->|Sí (OPTIMAL)| Z["Retornar PipelineResult (fallback=False)"]
    F -->|No (INFEASIBLE)| G{¿Hay candidatos alternativos?}
    G -->|Sí| H["Fallback: solve_joint(inst)"]
    H --> I{¿res_joint.solved?}
    I -->|Sí| J["Actualizar instancia con lecturas óptimas"]
    J --> K["Retornar PipelineResult (fallback=True)"]
    I -->|No| L["Retornar PipelineResult (INFEASIBLE)"]
    G -->|No| L
```

1. **Intento Primario (Rápido):** Resuelve la instancia asumiendo que la lectura top-1 de la visión es correcta mediante el modelo aritmético (Variante A). En condiciones normales de iluminación y nitidez, este paso resuelve el puzzle en menos de 15 ms.
2. **Fallback Automático (Resiliente):** Si el intento primario devuelve `INFEASIBLE` y la visión registró candidatos alternativos (`inst.candidates`), el pipeline conmuta automáticamente a la inferencia conjunta (Variante C). Maximiza la verosimilitud de las lecturas corrigiendo caracteres dudosos y entregando la solución correcta.

### 5.3. Estructura de Datos `PipelineResult`

```python
@dataclass
class PipelineResult:
    image: np.ndarray          # Imagen original en formato BGR
    structure: Structure       # Esquinas, homografía H, n, coordenadas y jaulas
    instance: Instance         # Instancia KenKen (actualizada si hubo fallback)
    solve_result: SolveResult  # Solución, status, tiempo y ramas del solver CP
    fallback_used: bool = False # Indicador de activación de inferencia conjunta
```

### 5.4. Uso en Código y CLI

**Desde Python:**
```python
from kenken import solve_image

result = solve_image("dataset/real/puzzle_01.jpg", method="auto")

if result.solved:
    print(f"KenKen {result.structure.n}x{result.structure.n} resuelto con éxito!")
    print(f"Fallback conjunto activado: {result.fallback_used}")
    for row in result.grid:
        print(row)
```

**Desde la Línea de Comandos:**
```bash
python -m kenken dataset/real/puzzle_01.jpg
```

---

### 5.5. Validación y Pruebas Unitarias

Pruebas implementadas en [`tests/test_pipeline.py`](../tests/test_pipeline.py):
* `test_solve_image_synthetic_end_to_end`: Ejecuta el flujo completo desde el render sintético vectorial hasta la solución, comprobando que `res.grid == gt["solution"]`.
* `test_solve_image_methods`: Valida la invocación explícita con `method="arithmetic"`, `method="table"` y `method="joint"`.
* `test_solve_image_auto_triggers_fallback`: Simula un fallo forzado en la lectura top-1 de una jaula, verificando que el modo `"auto"` activa exitosamente el fallback (`fallback_used == True`) y recupera la solución correcta sin intervención.
* `test_solve_image_file_not_found` y `test_solve_image_invalid_method`: Valida el control de excepciones ante archivos inexistentes o argumentos inválidos.

**Resultado de ejecución de pruebas:**
```bash
python -m pytest tests/test_pipeline.py -q
# 7 passed in 2.84s (100% éxito)
```

---

## 6. Benchmarking Experimental y Análisis de Complejidad

### 6.1. Metodología de Evaluación

El módulo [`kenken/benchmark.py`](../kenken/benchmark.py) evalúa el rendimiento computacional sistemático del solver variando:
* **Dimensión de la grilla:** $n \in \{3, 4, 5, 6, 7, 8, 9\}$.
* **Variantes de modelado:**
  1. `A_aritmética`: Descomposición intensional.
  2. `A_redundante`: Aritmética + sumas lineales de fila/columna ($\sum = n(n+1)/2$).
  3. `B_tabla`: Restricción global de tabla `AddAllowedAssignments` (GAC).
  4. `B_redundante`: Tabla + sumas lineales redundantes.
* **Métricas registradas:** Tiempo de CPU (*Wall time* en ms), ramas del árbol de búsqueda (*Branches*), conflictos resueltos (*Conflicts*) y verificación de exactitud.

### 6.2. Resultados Experimentales

Tabla consolidada a partir de [`results/cp_benchmark.csv`](../results/cp_benchmark.csv):

| $n$ | Tamaño Grilla | Espacio Bruto $n^{n^2}$ | Variante A (ms) | Variante A + Redundante (ms) | Variante B Tabla (ms) | Variante B + Redundante (ms) | Ramas Medias | Conflictos |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | $3 \times 3$ | $1.97 \times 10^4$ | $9.07$ | $15.26$ | $15.00$ | $14.47$ | 0 | 0 |
| 4 | $4 \times 4$ | $4.29 \times 10^9$ | $10.38$ | $14.28$ | $15.46$ | $14.70$ | 0 | 0 |
| 5 | $5 \times 5$ | $2.98 \times 10^{17}$ | $13.53$ | $14.56$ | $14.51$ | $14.46$ | 0 | 0 |
| 6 | $6 \times 6$ | $1.03 \times 10^{28}$ | $9.54$ | $14.39$ | $15.42$ | $14.22$ | 0 | 0 |
| 7 | $7 \times 7$ | $2.56 \times 10^{41}$ | $9.23$ | $14.56$ | $13.68$ | $14.26$ | 0 | 0 |
| 8 | $8 \times 8$ | $6.28 \times 10^{57}$ | $12.68$ | $14.31$ | $13.25$ | $12.18$ | 0 | 0 |
| 9 | $9 \times 9$ | $1.96 \times 10^{77}$ | $20.03$ | $13.03$ | $12.11$ | $17.22$ | 0 | 0 |

### 6.3. Análisis de Complejidad y Discusión Teórica (para el Informe IEEE)

1. **Espacio de Búsqueda Combinatorio vs. Búsqueda Real:**
   - El espacio de estados de fuerza bruta crece como $\mathcal{O}(n^{n^2})$. Para $n=9$, existen $\approx 1.96 \times 10^{77}$ configuraciones posibles (un orden de magnitud comparable con el número de átomos en el universo observable).
   - Sin embargo, las ramas exploradas por el solver CP-SAT son **$0$ en prácticamente todas las instancias**. Esto demuestra que la combinación de restricciones globales `AllDifferent` con las restricciones de jaula permite deducir la solución directamente en el **nodo raíz** (*root node deduction*) mediante consistencia de arco y *bounds propagation*, sin requerir ramificación exploratoria (*backtracking*).
2. **Impacto de las Restricciones de Tabla (Variante B):**
   - Para $n=9$, la Variante B resolvió en **$12.11\text{ ms}$**, superando a la Variante A sin redundancia ($20.03\text{ ms}$). La consistencia de arco generalizada (GAC) sobre el scope completo de las jaulas poda valores imposibles de forma más agresiva en problemas de mayor dimensión.
3. **Efecto de Restricciones Redundantes:**
   - Para tableros pequeños ($n \le 4$), agregar la suma redundante $\sum = n(n+1)/2$ agrega una ligera sobrecarga de propagación ($\sim 5\text{ ms}$). Para tableros mayores ($n=9$), la restricción redundante acelera la variante aritmética de $20.03\text{ ms}$ a $13.03\text{ ms}$.

### 6.4. Figuras Generadas

La figura [`results/figs/cp_benchmark.png`](../results/figs/cp_benchmark.png) resume gráficamente estos hallazgos en tres paneles de alta resolución:
1. Curvas de tiempo de resolución vs $n$ por variante.
2. Contraste entre el espacio de búsqueda teórico ($\log_{10}(n^{n^2})$) y el número de ramas reales exploradas.
3. Gráfico de barras comparativo por dimensión de tablero.

**Comando de regeneración:**
```bash
python -m kenken.benchmark --sizes 3 4 5 6 7 8 9 --repeats 3
```

---

## 7. Formalización en LaTeX y Estructura del Informe Técnico

### 7.1. Estructura de Archivos del Informe

Los archivos del informe técnico final en formato **IEEEtran** se encuentran organizados en la carpeta [`informe/`](../informe/):

* [`informe/main.tex`](../informe/main.tex): Documento raíz en formato IEEEtran. Incluye título, afiliación institucional (UPC), resumen ejecutivo en español, introducción con fundamentación neuro-simbólica, arquitectura del pipeline end-to-end, inclusión modular de la sección de CP, conclusiones y comandos de bibliografía.
* [`informe/cp_model_section.tex`](../informe/cp_model_section.tex): Sección modular que detalla con rigor matemático:
  - Definición formal del CSP como la tupla $\mathcal{P} = \langle X, D, C \rangle$.
  - Variables de decisión $x_{i,j}$ y dominios $D(x_{i,j}) = \{1, \dots, n\}$.
  - Restricciones globales de Cuadrado Latino ($\text{AllDifferent}$ en filas y columnas) y restricciones redundantes de suma triangular $\sum = n(n+1)/2$.
  - Formulación de Jaulas bajo la **Variante A** (Aritmética intensional con descomposición encadenada de productos y división reificada vía variable booleana $b_{\text{dir}}$).
  - Formulación de Jaulas bajo la **Variante B** (Restricciones globales de tabla extensional con tuplas factibles $\mathcal{T}_c$ y garantía de Consistencia de Arco Generalizada - GAC).
  - Formulación de la **Variante C** (Inferencia conjunta neuro-simbólica MAP mediante variables de hipótesis $r_{c,k}$, $\text{ExactlyOne}$, reificación condicional con $\text{OnlyEnforceIf}$ y función objetivo ponderada por log-probabilidades).
  - Análisis de complejidad teórica ($\mathcal{O}(n^{n^2})$ vs. reducción por Cuadrados Latinos y deducción en el nodo raíz vía CP-SAT / LCG).
  - Cuadro consolidado de resultados experimentales y figura comparativa de rendimiento.
* [`informe/refs.bib`](../informe/refs.bib): Base de datos bibliográfica BibTeX con referencias académicas primarias:
  - Jean-Charles Régin (AAAI 1994) para el filtrado de `AllDifferent`.
  - Christian Bessière et al. (Constraints 2006) para algoritmos de filtrado y GAC en restricciones de tabla.
  - Olga Ohrimenko, Peter Stuckey et al. (CP 2009) para *Lazy Clause Generation* (LCG).
  - Laurent Perron & Vincent Furnon (Google OR-Tools 2024) para CP-SAT.
  - Luc De Raedt et al. (AAAI 2011) para integración de programación por restricciones y aprendizaje automático.
  - Gary Bradski (2000) para la biblioteca OpenCV.
* [`informe/figs/cp_benchmark.png`](../informe/figs/cp_benchmark.png): Imagen del benchmark en tres paneles lista para compilación autosuficiente.

### 7.2. Compilación del Documento

El documento está diseñado para compilar de manera directa tanto en plataformas en la nube (**Overleaf**) como en entornos locales con **TeX Live** o **MiKTeX**:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 7.3. Cobertura de Criterios de la Rúbrica del Trabajo Práctico (CP)

| Componente de Rúbrica | Ponderación | Implementación en Código | Documentación y Reporte |
|:---|:---:|:---|:---|
| **Restricciones Globales** | 3 pts | `AddAllDifferent`, `AddAllowedAssignments` en [`kenken/model.py`](../kenken/model.py) | Sección 2, 3 y [`informe/cp_model_section.tex`](../informe/cp_model_section.tex) Sec. II-B, II-D |
| **Restricciones Reificadas** | 1 pt | `OnlyEnforceIf` para división no conmutativa y para hipótesis OCR en Variante C | Sección 4 y [`informe/cp_model_section.tex`](../informe/cp_model_section.tex) Sec. II-C.5, II-E |
| **Modelado Formal CSP** | 3 pts | Especificación formal $\langle X, D, C \rangle$, dominios, variantes A, B, C | Sección 1 a 4 y [`informe/cp_model_section.tex`](../informe/cp_model_section.tex) Sec. II |
| **Integración Pipeline** | 2 pts | `solve_image()` con fallback automático a inferencia conjunta en [`kenken/pipeline.py`](../kenken/pipeline.py) | Sección 5 e [`informe/main.tex`](../informe/main.tex) Sec. II |
| **Informe Técnico** | 5 pts | Código LaTeX completo, modular y listo para compilar con bibliografía | [`informe/main.tex`](../informe/main.tex), [`informe/cp_model_section.tex`](../informe/cp_model_section.tex), [`informe/refs.bib`](../informe/refs.bib) |
