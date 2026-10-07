# Plan de implementación — TB1: KenKen Solver (Visión Computacional + Constraint Programming)

Puzzle elegido: **KenKen** (continúa el documento de avance "KenKen avance.docx").
Base técnica: material del curso (OR-Tools CP-SAT: `AddAllDifferent` / cuadrado latino de la semana 3,
reificación con `OnlyEnforceIf` y `AddMultiplicationEquality` de la semana 5, `SolutionCallback` para
enumerar soluciones, y el patrón "extracción → JSON → modelo CP" de la semana 6 / LLM).
Inspiración: ejemplo de KU Leuven (Sudoku visual): en lugar de quedarse con la predicción más probable del
clasificador, el solver hace **inferencia conjunta**: elige la lectura más probable *que además sea
consistente con las reglas del puzzle*.

---

## 1. Estructura de la carpeta `TP/`

```
TP/
├── PLAN.md                  ← este archivo
├── README.md                ← instalación (requirements.txt) y ejecución exacta
├── requirements.txt         ← ortools, opencv-python, numpy, matplotlib, torch, torchvision, pillow, pandas
├── TP_KenKen.ipynb          ← notebook principal: demo end-to-end + experimentos + gráficos para el informe
├── kenken/                  ← código modular (lo que importa el notebook)
│   ├── __init__.py
│   ├── preprocessing.py     ← carga, gris, desenfoque, umbral adaptativo/Otsu, contorno del tablero, homografía
│   ├── grid.py              ← detección de n (tamaño) y coordenadas de celdas
│   ├── cages.py             ← grosor de bordes entre celdas adyacentes + Union-Find → jaulas
│   ├── ocr.py               ← recorte de la etiqueta, segmentación de caracteres, CNN, top-k lecturas
│   ├── cnn.py               ← arquitectura + entrenamiento del clasificador (PyTorch)
│   ├── instance.py          ← esquema JSON de la instancia + validación
│   ├── model.py             ← modelos CP-SAT (básico, tabla, inferencia conjunta)
│   ├── generator.py         ← generador aleatorio de instancias KenKen válidas (solución única)
│   ├── render.py            ← render sintético de tableros (imágenes con ground truth)
│   ├── visualize.py         ← grilla limpia + superposición de la solución sobre la foto original
│   ├── pipeline.py          ← solve_image(path) → instancia, solución, figura (sin intervención manual)
│   └── evaluate.py          ← métricas de visión, integración y solver
├── models/ocr_cnn.pt        ← pesos entrenados
├── dataset/
│   ├── real/                ← ≥10 imágenes (fotos impresas, ángulos, iluminación, capturas) + <nombre>.json (ground truth)
│   └── synthetic/           ← imágenes generadas automáticamente + JSON
├── results/                 ← CSV de métricas, tiempos, figuras
└── informe/                 ← LaTeX formato IEEE (IEEEtran): main.tex, refs.bib, figs/
```

## 2. Representación intermedia (puente Fase 1 → Fase 2)

```json
{
  "n": 4,
  "cages": [
    {"cells": [[0,0],[0,1]], "target": 3, "op": "-"},
    {"cells": [[0,2]],       "target": 2, "op": "="},
    {"cells": [[0,3],[1,3],[1,2]], "target": 24, "op": "*"}
  ],
  "candidates": {             // opcional: lecturas alternativas de la CNN (para inferencia conjunta)
    "0": [{"target": 3, "op": "-", "logp": -0.05}, {"target": 8, "op": "-", "logp": -3.2}]
  }
}
```

`instance.py` valida: toda celda pertenece a exactamente una jaula, jaulas conexas, `-`/`/` solo con 2 celdas,
target dentro de rangos alcanzables (p. ej. suma ≤ n·|C|, producto divisible por combinaciones posibles).

## 3. Fase 1 — Visión Computacional e IA

| Paso | Técnica (OpenCV) | Salida |
|---|---|---|
| 1. Preprocesamiento | `cvtColor` gris, `GaussianBlur`, `adaptiveThreshold` (robusto a iluminación) y Otsu como alternativa | imagen binaria |
| 2. Tablero | `findContours` → contorno de mayor área → `approxPolyDP` (4 vértices) → ordenar esquinas | cuadrilátero |
| 3. Perspectiva | `getPerspectiveTransform` + `warpPerspective` a un cuadrado fijo (p. ej. 900×900); guardar `H` para proyectar la solución de vuelta | tablero rectificado |
| 4. Tamaño n | Perfiles de proyección horizontal/vertical de las líneas (morfología con kernels largos) → contar picos equiespaciados, n ∈ {3..9} | `n`, coordenadas de celdas |
| 5. Jaulas | Para cada frontera entre celdas vecinas, medir grosor/densidad de tinta en una franja centrada en la línea; separar "gruesa" vs "delgada" con umbral automático (Otsu/k-means 1D sobre esas medidas, por imagen) → Union-Find une celdas sin borde grueso | lista de jaulas |
| 6. Etiqueta | La etiqueta está en la esquina superior izquierda de la celda "primera" (fila mínima, luego columna mínima) de cada jaula: recortar esa zona, componentes conexas → caracteres ordenados por x | glifos |
| 7. Clasificación | CNN pequeña propia (PyTorch, 2–3 conv + FC) con 14 clases: `0-9 + - × ÷`. Entrenada con glifos renderizados en varias fuentes + aumentos (rotación, blur, ruido, grosor, perspectiva leve). Devuelve **probabilidades** por glifo | top-k lecturas por jaula con log-prob |

Notas:
- Celda de 1 sola casilla → operador `=` (sin símbolo).
- Si la jaula tiene ≥3 celdas el operador solo puede ser `+` o `×` (se usa como restricción en la inferencia conjunta).
- Tesseract queda como **baseline** de comparación en el informe (opcional si se logra instalar); la CNN es la solución principal.

## 4. Fase 2 — Modelo de Constraint Programming (OR-Tools CP-SAT)

**Variables y dominios:** `x[i][j] ∈ {1..n}` (`NewIntVar(1, n, ...)`), igual que el cuadrado latino de la semana 3.

**Restricciones globales:**
- `AddAllDifferent` por fila y por columna.
- Redundante (implicada): `sum(fila) == n(n+1)/2` — se mide si acelera.

**Restricciones de jaula (modelo parametrizado, se generan desde el JSON):**
| op | Restricción |
|---|---|
| `=` | `x == T` |
| `+` | `sum(cells) == T` (restricción lineal global) |
| `×` | `AddMultiplicationEquality` encadenado con variables intermedias |
| `−` | `AddAbsEquality(T, a - b)` |
| `÷` | **reificada**: `b → a == T·y`, `¬b → y == T·a` (`OnlyEnforceIf`), como en la semana 5 |

**Variante B (tabla):** cada jaula como `AddAllowedAssignments` con las tuplas precalculadas
(respetando que celdas de la jaula en la misma fila/columna sean distintas). Es otra restricción global; se
compara contra la variante A en el benchmark.

**Variante C — inferencia conjunta (idea KU Leuven):**
- Para cada jaula `c` y cada lectura candidata `k` (top-k de la CNN): booleano `r[c,k]`.
- `AddExactlyOne(r[c,:])` por jaula.
- La restricción aritmética de la lectura `k` se activa con `OnlyEnforceIf(r[c,k])` (reificación).
- Objetivo: `Maximize(Σ round(1000·logp[c,k]) · r[c,k])` → la lectura más probable que tenga solución.
- Esto corrige errores de OCR (p. ej. confundir `×` con `+` o `8` con `3`) y justifica con fuerza el
  criterio de **restricciones reificadas** de la rúbrica.

**Extras:** `SolutionCallback` para contar soluciones (verificar unicidad), y registrar
`WallTime()`, `NumBranches()`, `NumConflicts()`.

## 5. Fase 3 — Integración y visualización

- `pipeline.solve_image(path)` encadena todo sin intervención manual: imagen → JSON → modelo → solución.
- Si el modelo básico es INFEASIBLE, el pipeline cae automáticamente a la variante C con las lecturas alternativas.
- Visualización:
  1. Grilla limpia en matplotlib (jaulas con bordes gruesos, etiquetas y solución en otro color).
  2. Solución superpuesta sobre la **foto original**: dibujar dígitos en el tablero rectificado y proyectar con `H⁻¹`.
- Opcional: widget simple en el notebook (subir imagen → ver resultado).

## 6. Dataset y evaluación

- **Real (≥10, entregable):** capturas digitales de distintas webs/estilos, fotos de impresiones con
  iluminación desigual, sombra, ángulos leves, tamaños 4×4 a 7×7. Cada una con su JSON ground truth anotado a mano.
- **Sintético (cientos):** `generator.py` (cuadrado latino aleatorio + partición aleatoria en jaulas + operación
  compatible, se descarta si no tiene solución única) + `render.py` con fuentes/grosores variados, ruido,
  rotación, perspectiva. Sirve para entrenar/validar la CNN y medir visión a escala.
- **Métricas de visión:** exactitud de detección del tablero, de n, de agrupación de jaulas, de targets,
  de operadores, y % de instancias completas correctas (antes y después de la inferencia conjunta). Matriz de confusión de la CNN.
- **Métricas de integración:** % imagen→solución correcta sin intervención; fallos atribuibles a visión vs. modelo; tiempo total por etapa.
- **Solver:** tiempos vs. n (3..9) y número de jaulas sobre instancias generadas, comparando variantes A/B
  (+/− restricción redundante). Discusión de complejidad: espacio bruto n^(n²) vs. ramas reales exploradas.

## 7. Informe (LaTeX IEEE) y video

`informe/main.tex` con IEEEtran: Introducción, Trabajos relacionados (KU Leuven, CP-SAT, OCR),
Pipeline de visión (con figuras de cada etapa y métricas), Modelo formal CP (variables, dominios,
restricciones, reificación y por qué), Variante de inferencia conjunta, Experimentos (tablas/gráficos de
`results/`), Análisis de complejidad, Conclusiones. Se reutiliza la teoría ya escrita en el avance.
Video ≤5 min: cada integrante explica una fase + demo desde la foto hasta la solución.

## 8. Orden de trabajo (hitos)

1. **Esqueleto y modelo CP** — `instance.py`, `model.py` (variante A), `generator.py`; probar con 3–4 KenKen
   escritos a mano. *Hito: resuelve cualquier JSON válido.*
2. **Render sintético** — `render.py`; genera imágenes + JSON. *Hito: dataset sintético listo.*
3. **Preprocesamiento + grilla + jaulas** — evaluar sobre sintéticos y luego reales. *Hito: jaulas correctas en ≥90 % sintéticos.*
4. **CNN de caracteres** — entrenar con glifos sintéticos, evaluar en etiquetas recortadas de imágenes reales.
5. **Pipeline end-to-end + visualización** (con proyección sobre la foto original).
6. **Variantes B y C** del modelo + fallback automático.
7. **Dataset real (≥10) anotado + evaluación completa** → CSV y figuras en `results/`.
8. **Limpieza**, README, requirements, comentarios; **informe** y **video**.

## 9. Mapeo a la rúbrica

| Criterio (pts) | Cómo se cubre |
|---|---|
| Precisión de visión (3) | Pipeline OpenCV + CNN propia, métricas por subtarea |
| Condiciones distintas (1) | Umbral adaptativo, homografía, aumentos; dataset real variado |
| Formulación CP (3) | Modelo formal parametrizado desde JSON |
| Restricciones globales (3) | `AllDifferent`, suma lineal, `AllowedAssignments` (tabla), `ExactlyOne`, comparación empírica |
| Reificadas (1) | División reificada + inferencia conjunta con `OnlyEnforceIf` |
| Puente IA→CP automático (1) | `solve_image()` + fallback sin intervención manual |
| Visualización clara (1) | Grilla limpia + superposición sobre la foto |
| Código limpio (2) | Paquete `kenken/` modular, docstrings, README |
| Informe LaTeX (5) | IEEEtran con métricas y análisis de complejidad |
