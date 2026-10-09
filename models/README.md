# Modelos Preentrenados (`models/`)

Este directorio contiene los pesos de los modelos neuronales de visión por computador empleados en la etapa de reconocimiento óptico de caracteres (OCR) del solver.

---

## Modelos Disponibles

### 1. `ocr_cnn_finetuned.pt` (Recomendado / Por Defecto)
- **Descripción:** Red convolucional `GlyphCNN` ajustada finamente mediante **Focal Loss** ($\gamma=1.5$) y minería de ejemplos difíciles (*hard negative mining*).
- **Entrada:** Imágenes de un solo canal en escala de grises de dimensiones $32 \times 32$ píxeles, normalizadas a $[-1, 1]$.
- **Salida:** Distribución log-softmax sobre 14 clases (dígitos `0`-`9` y operadores `+`, `−`, `×`, `÷`).
- **Rendimiento:**
  - **99.51%** de exactitud sobre glifos sintéticos limpios.
  - **90.99%** de exactitud bajo degradaciones físicas severas (ruido, desenfoque gaussiano, cambios de iluminación y distorsión tipográfica).

### 2. `ocr_cnn.pt` (Modelo Base)
- **Descripción:** Checkpoint base entrenado únicamente con aumento de datos afín sobre fuentes tipográficas estándar.
- **Rendimiento:** **99.16%** de exactitud sobre glifos sintéticos limpios.

---

## Arquitectura (`GlyphCNN`)

```text
Entrada (1 x 32 x 32)
  │
  ├── Conv2D(1 -> 32, kernel=3, padding=1) + BatchNorm + ReLU
  ├── Conv2D(32 -> 32, kernel=3, padding=1) + BatchNorm + ReLU
  ├── MaxPool2D(2x2) + Dropout(0.1)
  │
  ├── Conv2D(32 -> 64, kernel=3, padding=1) + BatchNorm + ReLU
  ├── Conv2D(64 -> 64, kernel=3, padding=1) + BatchNorm + ReLU
  ├── MaxPool2D(2x2) + Dropout(0.2)
  │
  ├── Aplanado (Flatten -> 64 * 8 * 8 = 4096)
  ├── Linear(4096 -> 128) + BatchNorm + ReLU + Dropout(0.3)
  └── Linear(128 -> 14) + LogSoftmax
```

Para reentrenar o ajustar finamente cualquiera de estos modelos, consulte [`scripts/finetune.py`](../scripts/finetune.py).
