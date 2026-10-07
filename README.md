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

## Dataset sintético (hito 2)

```bash
python -m kenken.render --count 300 --out dataset/synthetic --seed 0
```

Genera pares `syn_XXXXX.(jpg|png)` + `syn_XXXXX.json` (≈1 min, ≈45 MB). La carpeta no se sube
a Git porque se regenera con ese comando. El JSON trae la instancia, la solución y, en `image`:
las 4 esquinas del tablero (TL, TR, BR, BL), la homografía, la caja normalizada de cada etiqueta,
el texto impreso y el estilo usado (fuente, grosores, glifos de operación).

Set de estrés (más inclinación, sombra, desenfoque, ruido y JPEG agresivo):

```bash
python -m kenken.render --count 150 --out dataset/synthetic_hard --seed 1 --level hard --photo-prob 1.0 --prefix hard
```

## Visión: tablero, grilla y jaulas (hito 3)

```bash
python -m kenken.evaluate dataset/synthetic results/structure_synthetic.csv
python -m kenken.evaluate dataset/synthetic_hard results/structure_synthetic_hard.csv
python -m kenken.evaluate dataset/synthetic_hard --no-labels    # ablación sin la regla de etiquetas
```

| Set | Imágenes | Tablero (<2 % error) | n | Jaulas exactas | Bordes |
|---|---|---|---|---|---|
| Normal | 300 | 100 % | 100 % | 100 % | 100 % |
| Difícil, sin regla de etiquetas | 150 | 100 % | 100 % | 96.0 % | 99.9 % |
| Difícil, con regla de etiquetas | 150 | 100 % | 100 % | 98.0 % | 99.8 % |

## OCR de etiquetas con CNN propia (hito 4)

Explicación detallada en [`docs/cnn_explicada.md`](docs/cnn_explicada.md).

```bash
python -m kenken.cnn                                    # genera ~47k glifos sintéticos, entrena (~20 min CPU)
python -m kenken.evaluate --ocr dataset/synthetic results/ocr_synthetic.csv
python -m kenken.evaluate --ocr dataset/synthetic_hard results/ocr_synthetic_hard.csv
python -m kenken.evaluate --ocr dataset/synthetic --no-alt   # ablación sin segmentaciones alternativas
python -m kenken.figures                                # curvas de entrenamiento y matrices de confusión
```

CNN: 3 bloques convolucionales + 2 capas lineales, 403k parámetros, entrada 32×32, 14 clases.
Validación 97.7 %; **99.5 % por glifo** en el set normal cuando la segmentación es correcta.

| Set | Segmentación | Etiqueta top-1 | Etiqueta top-5 | Tablero completo | Tablero (todas en top-5) |
|---|---|---|---|---|---|
| Normal, sin alternativas | 96.4 % | 95.8 % | 96.4 % | 71.0 % | 74.0 % |
| Normal, con alternativas | 96.4 % | 97.4 % | 98.9 % | 80.7 % | 88.7 % |
| Difícil, sin alternativas | 75.5 % | 68.1 % | 72.7 % | 18.0 % | 18.0 % |
| Difícil, con alternativas | 75.5 % | 74.9 % | 83.2 % | 27.3 % | 39.3 % |

"Tablero (todas en top-5)" es el techo que puede alcanzar la inferencia conjunta del modelo CP (hito 6).

## Estructura actual

| Archivo | Contenido |
|---|---|
| `kenken/instance.py` | Formato JSON de la instancia (`Instance`, `Cage`) y validación estructural |
| `kenken/model.py` | Modelo CP-SAT (variante A), conteo de soluciones, verificador independiente |
| `kenken/generator.py` | Generador de puzzles aleatorios con solución única |
| `kenken/render.py` | Render sintético: tablero limpio + simulación de foto (perspectiva, luz, sombra, ruido, JPEG) |
| `kenken/preprocessing.py` | Carga, binarización adaptativa/Otsu, detección del tablero, homografía |
| `kenken/grid.py` | Detección de n con perfiles de proyección y posición de las líneas |
| `kenken/cages.py` | Grosor de bordes + Otsu 1D + Union-Find; regla "una etiqueta por jaula" |
| `kenken/ocr.py` | Recorte de etiquetas, segmentación en caracteres, decodificación top-k con gramática |
| `kenken/cnn.py` | CNN propia (PyTorch): arquitectura, datos sintéticos, entrenamiento, inferencia |
| `kenken/pipeline.py` | `extract_structure(img)`: imagen → tablero, n, jaulas; `read_instance(st)`: → instancia + candidatas |
| `kenken/evaluate.py` | Métricas de estructura y de OCR por imagen → CSV |
| `kenken/figures.py` | Figuras del informe (curvas de entrenamiento, matrices de confusión) |
| `models/ocr_cnn.pt` | Pesos entrenados de la CNN |
| `docs/cnn_explicada.md` | Explicación detallada de la CNN y su entrenamiento |
| `kenken/visualize.py` | Figura de etapas de la visión |
| `examples/` | 3 instancias escritas a mano (3x3, 4x4, 5x5) + 1 generada (6x6, seed 6) |
| `tests/` | Pruebas con pytest |
