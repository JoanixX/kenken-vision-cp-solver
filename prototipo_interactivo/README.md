---
title: KenKen Vision-CP Solver
emoji: 🧩
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
license: mit
short_description: KenKen Solver con OpenCV, PyTorch CNN y OR-Tools CP-SAT
---

# 🧩 KenKen Vision-CP Solver (Streamlit App)

Sistema neuro-simbólico que combina **Visión Computacional** (OpenCV + GlyphCNN en PyTorch) con **Programación por Restricciones** (Google OR-Tools CP-SAT) para escanear y resolver acertijos lógicos KenKen en tiempo real.

## 🚀 Características

* **Captura con Cámara en Vivo:** Apunta tu webcam o la cámara de tu teléfono móvil al tablero de KenKen y toma una foto instantáneamente con `st.camera_input`.
* **Subida de Archivos:** Sube fotos, capturas de pantalla o imágenes escaneadas (.png, .jpg, .jpeg) mediante arrastrar y soltar.
* **Catálogo Demostrativo (21 Casos del TP):** Explora y resuelve con un solo clic los 21 tableros organizados en las 7 categorías técnicas del informe (Lectura Directa, Rescate MAP, Perspectiva, Gran Escala 6×6 a 9×9, Estrés Extremo, Avisos Preventivos e Infactibilidad).
* **Inferencia Conjunta Neuro-Simbólica:** Fallback automático mediante optimización MAP con restricciones reificadas (`OnlyEnforceIf`), deduciendo correcciones si el OCR presenta ambigüedad.
* **Visualización de Solución:**
  * **Comparativa Compuesta:** Muestra la entrada original con detección del tablero frente al resultado resuelto.
  * **Tablero Limpio Vectorial:** Gráfica nítida de alta resolución lista para publicación.
  * **Proyección sobre Foto:** Números proyectados en perspectiva sobre la imagen original ($H^{-1}$).
  * **Descarga 1-Click:** Botón integrado para descargar la imagen resuelta en formato PNG.

## 🛠️ Tecnologías

* **Streamlit:** Interfaz interactiva moderna con componentes nativos de cámara web y reactividad.
* **OpenCV:** Preprocesamiento morfológico, rectificación proyectiva homográfica ($H$) y segmentación de jaulas (*Union-Find*).
* **PyTorch:** Red `GlyphCNN` fine-tuneada con Focal Loss ($\gamma=1.5$) y minería de ejemplos difíciles para máxima resiliencia ante ruido y operadores ambiguos.
* **Google OR-Tools CP-SAT:** Motor de satisfacción y optimización de restricciones con *Lazy Clause Generation* (LCG) y Consistencia de Arco Generalizada (GAC).

## 💻 Ejecución Local

Para probar esta aplicación en tu propia máquina:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Iniciar la aplicación en Streamlit
streamlit run app.py
```

## 📖 Guía de Despliegue en la Nube

Consulta la guía paso a paso en [`DEPLOY_GUIDE.md`](DEPLOY_GUIDE.md) para desplegar esta aplicación en:
1. **Streamlit Community Cloud** (Recomendado, gratuito y conectado a GitHub).
2. **Hugging Face Spaces** (16 GB RAM gratis con SDK de Streamlit).
