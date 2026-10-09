---
title: KenKen Vision-CP Solver
emoji: 🧩
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: mit
short_description: KenKen Solver con OpenCV y OR-Tools CP-SAT
---

# 🧩 KenKen Vision-CP Solver (Hugging Face Space)

Sistema neuro-simbólico que combina **Visión Computacional** (OpenCV + CNN en PyTorch) con **Programación por Restricciones** (Google OR-Tools CP-SAT) para escanear y resolver acertijos lógicos KenKen en tiempo real.

## 🚀 Características

* **Captura con Cámara en Vivo:** Apunta tu webcam o la cámara de tu teléfono móvil al tablero de KenKen y toma una foto con un solo clic.
* **Subida de Imágenes:** Sube fotos, capturas de pantalla o imágenes escaneadas (.png, .jpg, .jpeg).
* **Galería de Ejemplos 1-Click:** Prueba tableros de órdenes $3 \times 3$, $4 \times 4$, $5 \times 5$, $6 \times 6$ y $9 \times 9$ instantáneamente.
* **Inferencia Conjunta Neuro-Simbólica:** Fallback automático mediante optimización MAP con restricciones reificadas (`OnlyEnforceIf`), corrigiendo posibles errores del OCR.
* **Visualización de Solución:**
  * **Comparativa Compuesta:** Muestra la entrada original con detección del tablero frente al resultado resuelto.
  * **Tablero Limpio Vectorial:** Gráfica nítida de alta resolución lista para publicación.
  * **Proyección sobre Foto:** Números proyectados en perspectiva sobre la imagen original ($H^{-1}$).

## 🛠️ Tecnologías

* **Gradio 6:** Interfaz interactiva y soporte de cámara web/móvil con HTTPS nativo.
* **OpenCV:** Preprocesamiento morfológico, rectificación proyectiva homográfica ($H$) y segmentación de jaulas (*Union-Find*).
* **PyTorch:** Red `GlyphCNN` fine-tuneada con Focal Loss ($\gamma=1.5$) y minería de ejemplos difíciles para máxima resiliencia ante ruido y operadores ambiguos.
* **Google OR-Tools CP-SAT:** Motor de satisfacción y optimización de restricciones con *Lazy Clause Generation* (LCG) y Consistencia de Arco Generalizada (GAC).

## 💻 Ejecución Local

Para probar esta aplicación en tu propia máquina:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Iniciar la aplicación
python app.py

# 3. Opcional: Generar enlace público temporal HTTPS para probar en tu celular:
python app.py --share
```

## 📖 Despliegue en Hugging Face Spaces

Consulta la guía detallada en [`DEPLOY_GUIDE.md`](DEPLOY_GUIDE.md) para aprender a crear y sincronizar tu propio Space gratuito en Hugging Face.
