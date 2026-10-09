# Datasets de Evaluación y Entrenamiento (`dataset/`)

Este directorio contiene las colecciones de imágenes y anotaciones ground truth utilizadas para evaluar las capacidades de visión computacional, OCR y resolución end-to-end.

---

## Conjuntos de Datos

### 1. `dataset/synthetic/`
- **Muestras:** 300 pares de archivos (imagen `.png`/`.jpg` + anotación `.json`).
- **Características:** Tableros generados sintéticamente con dimensiones $n \in \{3, \dots, 9\}$. Presentan variaciones de fuentes tipográficas, pequeñas inclinaciones angulares y texturas suaves simulando papel impreso.

### 2. `dataset/synthetic_hard/`
- **Muestras:** 150 pares de archivos (imagen + anotación `.json`).
- **Características:** Casos de alta dificultad visual que incorporan:
  - Ruido de sensor y artefactos de compresión JPEG.
  - Sombras intensas y gradientes de iluminación no homogéneos.
  - Perspectiva fotográfica inclinada y deformación homográfica pronunciada.
  - Glifos degradados con roturas de contorno y fuentes condensadas.

### 3. `dataset/challenge_stress/`
- **Muestras:** Conjunto de tableros de desafío seleccionados donde las lecturas Top-1 de la visión inducen contradicciones intencionales para evaluar el mecanismo de recuperación neuro-simbólico MAP.

---

## Esquema de Anotación Ground Truth (`.json`)

Cada imagen cuenta con un archivo JSON con la siguiente estructura:

```json
{
  "n": 4,
  "solution": [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [4, 1, 2, 3],
    [2, 3, 4, 1]
  ],
  "cages": [
    {
      "cells": [[0, 0], [0, 1]],
      "target": 3,
      "op": "+"
    },
    {
      "cells": [[0, 2]],
      "target": 3,
      "op": "="
    }
  ],
  "image": {
    "file": "syn_00000.png",
    "corners": [[45, 30], [550, 40], [560, 545], [40, 535]]
  }
}
```

Para generar nuevas muestras o datasets sintéticos, utilice [`scripts/make_challenge_dataset.py`](../scripts/make_challenge_dataset.py) o invoque `kenken.render.make_dataset()`.
