# Directorio de Resultados Experimentales (`results/`)

Este directorio almacena los datos cuantitativos, gráficos analíticos y registros generados por las suites de pruebas, benchmarking y entrenamiento del sistema **KenKen Vision-CP Solver**.

---

## Estructura

```text
results/
├── README.md               ← Índice y descripción de los artefactos
├── benchmarks/             ← Archivos CSV de métricas experimentales
│   ├── cp_benchmark.csv    ← Tiempos, ramas y conflictos por orden n (3 a 9)
│   ├── e2e_synthetic.csv   ← Evaluación end-to-end sobre dataset sintético normal
│   ├── e2e_synthetic_hard.csv ← Evaluación end-to-end con alta degradación
│   ├── ocr_*.csv           ← Evaluaciones de precisión y matrices de confusión OCR
│   └── structure_*.csv     ← Métricas de detección de bordes y jaulas
├── figs/                   ← Figuras y curvas en alta resolución
│   ├── cp_benchmark.png    ← Comparativa de variantes CP vs orden n
│   ├── cnn_training.png    ← Curvas de pérdida y precisión del modelo base
│   ├── cnn_finetuning_curves.png ← Curvas de entrenamiento de fine-tuning
│   ├── demo_solution.png   ← Ejemplo visual compuesto de resolución
│   └── ocr_*_confusion.png ← Matrices de confusión de clasificación de glifos
└── logs/                   ← Registros detallados de entrenamiento
    ├── cnn_training_log.txt
    └── cnn_finetuning_log.txt
```

> **Nota de compatibilidad:** Para garantizar compatibilidad con scripts históricos y ejecuciones directas previas, las copias en la raíz de `results/` se preservan junto con esta estructura modular.
