# Guía de Despliegue en la Nube: KenKen Vision-CP Solver (Streamlit)

Esta guía explica paso a paso cómo desplegar el prototipo interactivo de **Streamlit** en la nube de forma totalmente gratuita, permitiendo que cualquier persona (profesores, jurado o usuarios) pruebe el sistema desde su computadora o teléfono móvil con acceso a la cámara.

---

## 📌 Las 2 Mejores Opciones Gratuitas de Despliegue

| Característica | 🌟 Opción 1: Streamlit Community Cloud (Recomendada) | 🚀 Opción 2: Hugging Face Spaces |
|---|---|---|
| **Plataforma** | [share.streamlit.io](https://share.streamlit.io/) | [huggingface.co/spaces](https://huggingface.co/spaces) |
| **Integración** | Nativa directa con tu repositorio de GitHub | Repositorio Git independiente en HF |
| **Hardware Gratuito** | 1 vCPU / 1 GB RAM (Optimizado para Streamlit) | **2 vCPU / 16 GB RAM** |
| **HTTPS Nativo** | ✅ Sí (Permite cámara en celulares) | ✅ Sí (Permite cámara en celulares) |
| **Tiempo de Despliegue** | ~2 minutos | ~2 minutos |
| **Detección Automática** | Detecta `requirements.txt` y `packages.txt` | Detecta `README.md` (metadata) y `packages.txt` |

Ambas opciones son 100% compatibles con este proyecto. A continuación tienes el paso a paso para cada una.

---

## 🌟 Opción 1: Despliegue en Streamlit Community Cloud (Paso a Paso)

Esta es la opción más directa y cómoda porque se conecta automáticamente a tu repositorio en GitHub (`JoanixX/kenken-vision-cp-solver`) y se actualiza cada vez que haces `git push`.

### Paso 1: Subir tus cambios a GitHub
Asegúrate de que tus cambios locales estén guardados y subidos a la rama `main` de tu repositorio de GitHub:

```bash
git add .
git commit -m "feat: migrate interactive prototype to Streamlit"
git push origin main
```

### Paso 2: Conectar con Streamlit Community Cloud
1. Entra a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de **GitHub**.
2. En tu panel principal, haz clic en el botón **"Create app"** (o **"New app"**).
3. Selecciona la opción **"I already have an app"** (o conecta tu repositorio).
4. Configura los campos del formulario:
   * **Repository:** Selecciona `JoanixX/kenken-vision-cp-solver` (o tu usuario/repositorio).
   * **Branch:** `main`
   * **Main file path:** Escribe `streamlit_app.py` (o `prototipo_interactivo/app.py`).
   * **App URL:** (Opcional) Puedes personalizar el subdominio gratuito, por ejemplo:  
     `kenken-solver-upc.streamlit.app`
5. Haz clic en **"Deploy"**.

### Paso 3: Verificación del Despliegue
* Streamlit creará el contenedor en la nube, instalará las dependencias de `requirements.txt` y los paquetes de sistema de `packages.txt`.
* En aproximadamente 1 o 2 minutos tu aplicación estará en línea con HTTPS activo.
* ¡Listo! Comparte el enlace público en la entrega de tu Trabajo Parcial (TP).

---

## 🚀 Opción 2: Despliegue en Hugging Face Spaces (Paso a Paso)

Si prefieres usar la infraestructura de Hugging Face (que ofrece **16 GB de memoria RAM** en su capa gratuita), puedes desplegarlo en tu Space existente `joako2202/kenken-solver`.

### Estructura de la carpeta `prototipo_interactivo/`
Para Hugging Face, todo el código autocontenido se encuentra dentro de `prototipo_interactivo/`:

| Archivo / Carpeta | Propósito |
|---|---|
| `README.md` | Cabecera YAML que define el SDK como `streamlit` y el punto de entrada como `app.py`. |
| `app.py` | Aplicación interactiva de Streamlit con cámara, catálogo y métricas. |
| `requirements.txt` | Lista de librerías Python (`streamlit`, `ortools`, `opencv-python-headless`, etc.). |
| `packages.txt` | Paquetes de Linux Debian (`libgl1`, `libglib2.0-0`) para OpenCV. |
| `models/` | Pesos de la CNN (`ocr_cnn_finetuned.pt`). |
| `kenken/` | Pipeline neuro-simbólico y modelo CP-SAT. |
| `sample_images/` | Las 21 imágenes de prueba organizadas por categorías. |

---

### Método A: Despliegue mediante Git (Recomendado)

Desde tu terminal, dentro de la carpeta `prototipo_interactivo/`:

```bash
cd prototipo_interactivo

# 1. Verificar el remoto configurado
git remote -v
# Si no está configurado, agrégalo:
# git remote add space https://huggingface.co/spaces/joako2202/kenken-solver

# 2. Agregar los archivos actualizados y confirmar commit
git add .
git commit -m "feat: migrate prototype to Streamlit on Hugging Face Spaces"

# 3. Enviar a Hugging Face
git push space main
```

> **Autenticación en Hugging Face:**  
> Cuando Git te pida credenciales:  
> - **Username:** Tu usuario de Hugging Face (`joako2202`).  
> - **Password:** Usa un **User Access Token** con permisos de *Write* (lo generas en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

---

### Método B: Despliegue desde la Web de Hugging Face (Sin Terminal)

1. Abre tu Space en el navegador: [https://huggingface.co/spaces/joako2202/kenken-solver](https://huggingface.co/spaces/joako2202/kenken-solver).
2. Ve a la pestaña **Files and versions**.
3. Asegúrate de que `README.md` tenga la configuración:
   ```yaml
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
   ```
4. Sube o reemplaza `app.py` y `requirements.txt` haciendo clic en **Add file -> Upload files**.
5. Haz clic en **Commit changes to main**.
6. En la pestaña **App**, observa el build en los logs hasta que cambie a **Running**.

---

## 📱 Cómo Probar la Cámara en Smartphones

Gracias a que tanto Streamlit Cloud como Hugging Face Spaces cuentan con **certificado SSL (HTTPS)** nativo:
1. Abre el enlace público de tu app desde el navegador de tu celular (Google Chrome en Android, Safari en iOS).
2. Ve a la pestaña **"📸 Cámara en Vivo"**.
3. El navegador te solicitará permiso para acceder a la cámara; presiona **"Permitir"**.
4. Apunta al tablero de KenKen impreso y presiona **"Take Photo"**.
5. Presiona **"⚡ Resolver KenKen"** para ver la solución proyectada y las estadísticas en tiempo real.

---

## 💻 Ejecución y Verificación Local

Antes de desplegar, puedes verificar el funcionamiento en tu propia computadora:

```bash
# 1. Asegúrate de tener las dependencias instaladas
pip install -r prototipo_interactivo/requirements.txt

# 2. Ejecutar la aplicación
streamlit run prototipo_interactivo/app.py
```

La aplicación se abrirá automáticamente en tu navegador predeterminado en `http://localhost:8501`.

---

## 🔧 Solución de Problemas Frecuentes

### 1. ¿Por qué se utiliza `opencv-python-headless` y `packages.txt`?
En servidores en la nube basados en Linux Debian (como los de Streamlit Cloud o Hugging Face), OpenCV estándar busca librerías gráficas de escritorio como X11 y OpenGL. El archivo `packages.txt` con `libgl1` y `libglib2.0-0` junto con `opencv-python-headless` garantizan que no ocurra el error `ImportError: libGL.so.1: cannot open shared object file`.

### 2. ¿El modelo de PyTorch excede los límites de memoria?
No. La red `GlyphCNN` fine-tuneada pesa solo **1.6 MB** y está optimizada para inferencia en CPU consumiendo menos de 180 MB de memoria RAM total, lo que asegura un funcionamiento fluido tanto en máquinas con 1 GB de RAM como en Hugging Face (16 GB).

### 3. La cámara no abre en el celular
Verifica que estés ingresando mediante la URL con `https://` y no `http://`. Los navegadores modernos bloquean el acceso al hardware de cámara por motivos de seguridad si la conexión no está cifrada con SSL.
