# Scripts y Herramientas de Ejecución (`scripts/`)

Este directorio contiene las herramientas ejecutables para benchmarking, evaluación cuantitativa, entrenamiento/fine-tuning y análisis experimental del sistema **KenKen Vision-CP Solver**.

---

## 1. Herramientas Principales

### `benchmark.py`
Ejecuta el benchmark sistemático de rendimiento de Google OR-Tools CP-SAT a través de múltiples órdenes de grilla ($n \in [3, 9]$) comparando:
- **Variante A:** Aritmética intensional canónica.
- **Variante A + Redundante:** Restricciones implicadas de suma de Cuadrado Latino.
- **Variante B:** Restricciones globales de tabla (`AddAllowedAssignments` / GAC).
- **Variante B + Redundante.**

```bash
# Ejecución estándar (3 repeticiones por orden n)
python scripts/benchmark.py --sizes 3 4 5 6 7 8 9 --repeats 3

# Personalizar rutas de salida
python scripts/benchmark.py --out-csv results/benchmarks/mi_benchmark.csv --out-fig results/figs/mi_grafica.png
```

---

### `evaluate.py`
Evalúa cuantitativamente los diferentes componentes del pipeline frente a datasets con ground truth JSON:
- `--structure`: Mide precisión de tablero, tamaño $n$, $F_1$ de jaulas y clasificación de bordes.
- `--ocr`: Mide exactitud Top-1 y Top-$k$ de caracteres y genera matrices de confusión.
- `--e2e`: Evalúa el flujo completo desde la imagen hasta la solución verificada por el solver.

```bash
# Evaluar segmentación geométrica de jaulas
python scripts/evaluate.py dataset/synthetic --structure

# Evaluar reconocimiento óptico de caracteres (OCR)
python scripts/evaluate.py dataset/synthetic --ocr

# Evaluar pipeline completo end-to-end
python scripts/evaluate.py dataset/synthetic --e2e
```

---

### `finetune.py`
Realiza el fine-tuning de la red convolucional `GlyphCNN` utilizando **Focal Loss** ($\gamma = 1.5$) y minería de ejemplos difíciles para resolver ambigüedades comunes (+ vs *, 8 vs 5, etc.):

```bash
python scripts/finetune.py --epochs 8 --lr 0.0003
```

---

### `generate_figures.py`
Genera las curvas de convergencia y las matrices de confusión listas para el informe técnico y publicaciones:

```bash
python scripts/generate_figures.py
```

---

### `make_challenge_dataset.py`
Genera instancias sintéticas diseñadas específicamente para inducir incertidumbre visual y poner a prueba la formulación de inferencia conjunta neuro-simbólica (MAP):

```bash
python scripts/make_challenge_dataset.py --count 50 --out dataset/challenge_stress
```

---

## 2. Experimentos y Estudios (`scripts/experiments/`)

- [`comprehensive_map_study.py`](experiments/comprehensive_map_study.py): Análisis global de recuperación de fallos y rescate MAP a lo largo de conjuntos completos de imágenes sintéticas.
- [`test_map_analysis.py`](experiments/test_map_analysis.py): Traza profunda de candidatos y log-probabilidades para muestras individuales desafiantes.
