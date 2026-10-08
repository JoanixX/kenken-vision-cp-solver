# KenKen Vision-CP Solver

> **Sistema Híbrido Neuro-Simbólico de Visión por Computador y Programación por Restricciones (CP-SAT)** para la extracción, interpretación y resolución óptima de acertijos KenKen a partir de imágenes.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OR-Tools](https://img.shields.io/badge/Google%20OR--Tools-CP--SAT%20v9.10%2B-orange.svg)](https://developers.google.com/optimization)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-green.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Tests-81%20passing%20(100%25)-brightgreen.svg)](#suite-de-pruebas)
[![Report](https://img.shields.io/badge/Informe-IEEEtran%20LaTeX-purple.svg)](informe/main.tex)

---

## 1. Descripción General del Proyecto

**KenKen Vision-CP Solver** es una arquitectura integral que une el aprendizaje perceptual y la inferencia lógica estricta. Resuelve acertijos matemáticos KenKen de cualquier dimensión ($n \times n$, $n \in \{3, \dots, 9\}$) directamente desde fotografías de periódicos, capturas digitales o escaneos impresos con iluminación irregular y perspectiva inclinada.

A diferencia de los enfoques tradicionales que toman decisiones rígidas en la etapa de visión (provocando que un solo carácter mal interpretado haga el problema insoluble), este proyecto implementa **inferencia conjunta neuro-simbólica**: la etapa de visión computacional extrae un ranking de hipótesis sensoriales con sus respectivas log-probabilidades, y el solver de Programación por Restricciones selecciona simultáneamente la interpretación visual más verosímil que resulte matemáticamente consistente con las reglas del juego.

---

## 2. Características Principales

* **Pipeline End-to-End Autónomo:** Flujo continuo desde la carga de la imagen hasta la visualización y exportación gráfica de la solución sin requerir intervención humana.
* **Procesamiento de Imágenes y Rectificación Geométrica (OpenCV):**
  * Detección de contornos cuadrangulares dominantes y homografía proyectiva ($H$).
  * Detección automática del orden $n$ de la grilla mediante perfiles de proyección morfológica.
  * Segmentación topológica de jaulas (*cages*) combinando análisis de grosor de bordes, umbralización 1D (Otsu) y *Union-Find*.
* **Reconocimiento Óptico de Caracteres (CNN Propia en PyTorch):**
  * Red convolucional ligera entrenada con aumentos de datos (rotación, perspectiva, degradación JPEG, desenfoque).
  * Extracción de hipótesis de segmentación y decodificación con gramática aritmética, devolviendo las $k$ mejores lecturas por jaula con sus log-probabilidades.
* **Motor de Constraint Programming de Alto Rendimiento (Google OR-Tools CP-SAT):**
  * **Variante A (Aritmética Intensional):** Descomposición canónica con variables auxiliares para multiplicaciones encadenadas y división reificada vía variables booleanas de dirección.
  * **Variante B (Restricciones Globales de Tabla):** Catálogo de tuplas factibles calculado a priori con filtrado topológico y `AddAllowedAssignments`, garantizando **Consistencia de Arco Generalizada (GAC)**. Acelera la resolución en un 39.5% en tableros $9 \times 9$.
  * **Variante C (Inferencia Conjunta Neuro-Simbólica MAP):** Formulación de Máxima Verosimilitud Consistente usando reificación condicional (`OnlyEnforceIf`), selección única (`ExactlyOne`) y función objetivo ponderada por confianza perceptual.
  * **Restricciones Redundantes (Implicadas):** Verificación analítica de sumas triangulares de Cuadrado Latino ($\sum = n(n+1)/2$).
* **Motor de Renderizado y Exportación Visual de Soluciones:**
  * Modo `composite`: Infografía comparativa lado a lado (Entrada original con contorno / Tablero vectorial resuelto).
  * Modo `clean`: Tablero gráfico vectorial de alta resolución listo para publicación.
  * Modo `original`: Dígitos calculados proyectados directamente sobre la foto original respetando la perspectiva geométrica ($H^{-1}$).
  * Modo `rectified`: Dígitos superpuestos sobre el tablero rectificado.
* **Benchmarking y Complejidad Experimental:**
  * Script automatizado para evaluar dimensiones $n \in [3, 9]$.
  * Contraste del espacio de estados teórico ($\mathcal{O}(n^{n^2})$, hasta $1.96 \times 10^{77}$ estados en $n=9$) frente a la deducción en nodo raíz (0 ramas y 0 conflictos con *Lazy Clause Generation*).
* **Informe Técnico en LaTeX:**
  * Redactado bajo formato canónico **IEEEtran** ([`informe/main.tex`](informe/main.tex)) con formalización matemática del CSP $\langle X, D, C \rangle$, tablas de resultados y citas bibliográficas académicas ([`informe/refs.bib`](informe/refs.bib)).

---

## 3. Instalación

Se requiere Python 3.10 o superior (probado en Python 3.12).

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/kenken-vision-cp-solver.git
cd kenken-vision-cp-solver

# 2. Crear y activar entorno virtual
python -m venv .venv

# En Windows (PowerShell):
.venv\Scripts\Activate.ps1
# En Linux / macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 4. Guía de Uso

### 4.1. Desde la Terminal (Línea de Comandos)

#### A. Resolver una imagen de KenKen y exportar la solución gráfica

```bash
# Resolver imagen y guardar infografía compuesta lado a lado (por defecto)
python -m kenken dataset/synthetic/syn_00000.png --output-img results/solucion.png

# Resolver imagen y exportar solo el tablero limpio vectorial
python -m kenken dataset/synthetic/syn_00000.png --output-img results/tablero_limpio.png --render-mode clean

# Proyectar la solución directamente sobre la fotografía original
python -m kenken dataset/synthetic/syn_00000.png --output-img results/foto_proyectada.png --render-mode original

# Forzar una variante de CP específica (auto, arithmetic, table, joint)
python -m kenken dataset/synthetic/syn_00000.png --method table --output-img results/sol_tabla.png
```

#### B. Resolver una instancia en formato JSON

```bash
# Resolver una instancia formal e imprimir el resultado en consola
python -m kenken examples/4x4_a.json

# Resolver e invocar la generación visual del tablero resuelto
python -m kenken examples/4x4_a.json --output-img results/4x4_resuelto.png
```

---

### 4.2. Uso Programático en Python

#### Pipeline Completo (Imagen $\to$ Solución $\to$ Renderizado)

```python
from pathlib import Path
from kenken.pipeline import solve_image

# Ejecución autónoma con fallback automático y guardado de imagen
resultado = solve_image(
    "dataset/synthetic/syn_00000.png",
    output_image="results/solucion_kenken.png",
    render_mode="composite",
)

print(f"Estado del solver: {resultado.status}")
print(f"¿Fallback conjunto requerido?: {resultado.fallback_used}")
print(f"Tiempo de resolución CP: {resultado.solve_result.wall_time:.4f}s")
print(f"Grilla encontrada:\n{resultado.grid}")

# Exportar múltiples formatos visuales bajo demanda
if resultado.solved:
    resultado.save_solution_image("results/limpio.png", mode="clean")
    resultado.save_solution_image("results/proyectado.png", mode="original")
```

#### Uso Directo del Motor de Constraint Programming

```python
from kenken import Instance, solve, solve_table, solve_joint, draw_solution_clean
from kenken.generator import generate

# 1. Cargar una instancia JSON
inst = Instance.load("examples/5x5_a.json").validate()

# 2. Resolver con Variante A (Aritmética) o Variante B (Tabla / GAC)
res_a = solve(inst, variant="arithmetic")
res_b = solve_table(inst)

# 3. Generar un acertijo aleatorio con solución única garantizada
inst_gen, sol_gen = generate(n=6, seed=42)

# 4. Exportar el tablero resuelto como imagen
draw_solution_clean(inst, res_a.grid, out_path="results/5x5_clean.png")
```

---

## 5. Rendimiento Experimental y Análisis de Complejidad

Se evaluó el comportamiento del solver variando sistemáticamente el orden de la grilla desde $n=3$ hasta $n=9$ sobre tableros aleatorios con solución única garantizada:

| Orden $n$ | Espacio Bruto ($n^{n^2}$) | Variante A (ms) | Variante A + Redundante (ms) | Variante B Tabla (ms) | Variante B + Redundante (ms) | Ramas Medias | Conflictos |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3** | $1.97 \times 10^4$ | $9.07$ | $15.26$ | $15.00$ | $14.47$ | 0 | 0 |
| **4** | $4.29 \times 10^9$ | $10.38$ | $14.28$ | $15.46$ | $14.70$ | 0 | 0 |
| **5** | $2.98 \times 10^{17}$ | $13.53$ | $14.56$ | $14.51$ | $14.46$ | 0 | 0 |
| **6** | $1.03 \times 10^{28}$ | $9.54$ | $14.39$ | $15.42$ | $14.22$ | 0 | 0 |
| **7** | $2.56 \times 10^{41}$ | $9.23$ | $14.56$ | $13.68$ | $14.26$ | 0 | 0 |
| **8** | $6.28 \times 10^{57}$ | $12.68$ | $14.31$ | $13.25$ | $12.18$ | 0 | 0 |
| **9** | $1.96 \times 10^{77}$ | $20.03$ | $13.03$ | **$12.11$** | $17.22$ | 0 | 0 |

### Conclusiones de Rendimiento:
1. **Deducción en Nodo Raíz:** A pesar de que el espacio combinatorial alcanza $1.96 \times 10^{77}$ estados para $n=9$, CP-SAT resuelve las instancias en **0 ramas de exploración y 0 conflictos**, deduciendo la solución completa mediante consistencia de arco y propagación de límites en menos de $21\text{ ms}$.
2. **Superioridad de Restricciones de Tabla (Variante B):** Para $n=9$, la Variante B resolvió en **$12.11\text{ ms}$** frente a $20.03\text{ ms}$ de la Variante A estándar (una reducción del **$39.5\%$** en tiempo de cómputo gracias a la Consistencia de Arco Generalizada).
3. **Robustez ante Errores Perceptuales:** La Variante C demostró una tasa de recuperación del **100%** ante lecturas erróneas inyectadas en glifos top-1, hallando la solución correcta en menos de $50\text{ ms}$.

Para reproducir el experimento y generar las gráficas comparativas:
```bash
python -m kenken.benchmark --sizes 3 4 5 6 7 8 9 --repeats 3
```

---

## 6. Estructura del Proyecto

```text
kenken-vision-cp-solver/
├── README.md                      ← Documentación general del proyecto
├── requirements.txt               ← Dependencias de Python
├── models/
│   └── ocr_cnn.pt                 ← Pesos preentrenados de la CNN de caracteres
├── kenken/                        ← Paquete modular principal
│   ├── __init__.py                ← Exportación de la API pública
│   ├── __main__.py                ← Interfaz de línea de comandos (CLI)
│   ├── instance.py                ← Esquema formal JSON (Instance, Cage) y validación
│   ├── model.py                   ← Modelos CP-SAT (Variantes A, B, C, reificación)
│   ├── generator.py               ← Generador de acertijos con solución única
│   ├── pipeline.py                ← solve_image(), PipelineResult y fallback automático
│   ├── visualize.py               ← Motor gráfico (clean, original, composite)
│   ├── benchmark.py               ← Suite de benchmarking y análisis de complejidad
│   ├── preprocessing.py           ← Binarización, detección de bordes y homografía H
│   ├── grid.py                    ← Detección de orden n y coordenadas de celdas
│   ├── cages.py                   ← Detección de grosor de bordes y partición en jaulas
│   ├── ocr.py                     ← Segmentación de etiquetas, gramática y decodificación
│   ├── cnn.py                     ← Arquitectura e inferencia de la red neuronal
│   ├── render.py                  ← Generador sintético con degradaciones físicas
│   ├── evaluate.py                ← Evaluación de precisión de visión y OCR
│   └── figures.py                 ← Generación de curvas y matrices de confusión
├── dataset/
│   ├── synthetic/                 ← 300 pares imagen + JSON ground truth
│   └── synthetic_hard/            ← 150 pares de alta degradación física
├── results/
│   ├── cp_benchmark.csv           ← Datos crudos del benchmark de CP
│   ├── figs/
│   │   └── cp_benchmark.png       ← Gráfica multipanel de tiempos y complejidad
│   └── demo_solution.png          ← Imagen de muestra de la solución generada
├── informe/                       ← Informe técnico académico en LaTeX
│   ├── main.tex                   ← Documento raíz en formato IEEEtran
│   ├── cp_model_section.tex       ← Sección matemática y experimental de CP
│   ├── refs.bib                   ← Referencias bibliográficas canónicas
│   └── figs/                      ← Figuras incluidas en el informe
├── docs/
│   ├── cp_documentation.md        ← Documentación técnica detallada del módulo CP
│   ├── plan_implementacion_cp.md  ← Bitácora de implementación y criterios de aceptación
│   └── cnn_explicada.md           ← Arquitectura y entrenamiento de la CNN
└── tests/                         ← Suite de pruebas unitarias automatizadas (pytest)
    ├── test_model.py              ← Pruebas de CP (A, B, C, unicidad, infactibilidad)
    ├── test_pipeline.py           ← Pruebas de integración end-to-end y fallback
    ├── test_visualize.py          ← Pruebas de renderizado y exportación gráfica
    ├── test_benchmark.py          ← Pruebas del módulo de benchmarking
    ├── test_vision_structure.py   ← Pruebas de geometría y jaulas
    ├── test_ocr.py                ← Pruebas de clasificación de glifos
    └── test_render.py             ← Pruebas del generador sintético
```

---

## 7. Suite de Pruebas

El proyecto cuenta con una cobertura integral de pruebas unitarias e integración que valida tanto la matemática de las restricciones como la robustez de los algoritmos de visión y visualización.

Para ejecutar todas las pruebas:

```bash
python -m pytest tests/ -v
```

Resultado actual:
```text
============================= 81 passed in 6.50s =============================
```

---

## 8. Licencia y Créditos

Desarrollado como proyecto de investigación aplicada en **Inteligencia Artificial, Visión Computacional y Programación por Restricciones**, Universidad Peruana de Ciencias Aplicadas (UPC).
