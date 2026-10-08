# Guía de Despliegue en Hugging Face Spaces: KenKen Vision-CP Solver

Esta guía explica paso a paso cómo funciona **Hugging Face Spaces** y cómo desplegar este prototipo interactivo para que cualquier persona pueda probarlo desde su navegador o teléfono móvil con acceso a la cámara.

---

## 1. ¿Qué es Hugging Face Spaces y por qué usarlo?

**Hugging Face Spaces** es una plataforma en la nube diseñada para hospedar demostraciones interactivas de Machine Learning e Inteligencia Artificial de forma gratuita.

### Ventajas principales:
* **Capa Gratuita Generosa:** Proporciona máquinas virtuales con **2 vCPU y 16 GB de memoria RAM** (suficiente para ejecutar modelos de visión por computador y solvers como CP-SAT).
* **HTTPS Nativo Automático:** Cada Space obtiene un subdominio público con certificado SSL (ej. `https://huggingface.co/spaces/tu-usuario/kenken-solver`). **Esto es indispensable para que los navegadores móviles (Chrome en Android, Safari en iOS) permitan acceder a la cámara física del celular.**
* **Despliegue Continuo con Git:** Cada Space es un repositorio Git estándar. Cualquier cambio que hagas localmente y envíes con `git push` se construye y despliega automáticamente.
* **Integración Nativa con Gradio:** Hugging Face mantiene y optimiza la librería Gradio, asegurando compatibilidad total con componentes de cámara (`webcam`), galerías de imágenes y visualizadores.

---

## 2. Anatomía de la Carpeta `prototipo_interactivo/`

Para que un Space funcione en Hugging Face, la carpeta debe contener los siguientes archivos clave:

| Archivo | Propósito en Hugging Face |
|---|---|
| `README.md` | **Obligatorio.** Contiene una cabecera YAML en las primeras líneas que le indica a Hugging Face qué SDK usar (`sdk: gradio`), qué versión y cuál es el archivo de entrada (`app_file: app.py`). |
| `app.py` | El archivo ejecutable principal que construye la interfaz con Gradio y llama al solver de KenKen. |
| `requirements.txt` | Lista de librerías Python que Hugging Face instalará con `pip` durante el build. |
| `packages.txt` | Lista de paquetes del sistema operativo Linux Debian que se instalarán con `apt-get` (por ejemplo, `libgl1` y `libglib2.0-0` necesarios para OpenCV). |
| `sample_images/` | Imágenes de prueba que se muestran en el componente `gr.Examples` para que cualquiera pueda probar el solver con un solo clic. |
| `models/ocr_cnn.pt` | Los pesos preentrenados de la red neuronal convolucional para el reconocimiento de caracteres. |
| `kenken/` | El código modular del pipeline de visión por computador y de los modelos de Constraint Programming. |

---

## 3. Paso a Paso: Cómo Desplegar en Hugging Face Spaces

### Paso 1: Crear tu cuenta y el Space en Hugging Face
1. Si no tienes cuenta, regístrate gratis en [huggingface.co](https://huggingface.co/).
2. Ve a [huggingface.co/new-space](https://huggingface.co/new-space).
3. Completa los campos:
   * **Space name:** `kenken-solver` (o el nombre que prefieras).
   * **License:** `mit` (o la que uses).
   * **Select the Space SDK:** Selecciona **Gradio**.
   * **Space hardware:** Selecciona **CPU basic · 2 vCPU · 16 GB · Free**.
   * **Privacy:** `Public` (para que cualquiera pueda usarlo).
4. Haz clic en **Create Space**.

---

### Paso 2: Subir tu código al Space

Hugging Face te proporcionará una URL de repositorio Git que luce así:
`https://huggingface.co/spaces/<tu-usuario>/kenken-solver`

Puedes subir el contenido de la carpeta `prototipo_interactivo/` usando cualquiera de estos dos métodos:

#### Método A: Mediante Git en tu Terminal (Recomendado)

Abre tu terminal en la carpeta `prototipo_interactivo/`:

```bash
cd prototipo_interactivo

# 1. Inicializar repositorio git local si aún no está inicializado
git init
git branch -M main

# 2. Agregar el repositorio remoto de Hugging Face
git remote add space https://huggingface.co/spaces/<tu-usuario>/kenken-solver

# 3. Agregar los archivos y hacer commit
git add .
git commit -m "feat: initial release of KenKen Vision-CP Solver Space"

# 4. Enviar al servidor de Hugging Face (te pedirá tu usuario y un Access Token de HF)
git push -u space main
```

> **Nota sobre autenticación:** Hugging Face te solicitará tu nombre de usuario y una contraseña. Como contraseña debes utilizar un **Access Token** (puedes crearlo gratis en tu perfil de Hugging Face en: *Settings -> Access Tokens -> New Token* con permisos de *Write*).

#### Método B: Mediante la Interfaz Web de Hugging Face (Sin usar comandos)
1. En la página de tu Space en Hugging Face, haz clic en la pestaña **Files and versions**.
2. Haz clic en **Add file -> Upload files**.
3. Arrastra los archivos y carpetas que están dentro de `prototipo_interactivo/` (`app.py`, `README.md`, `requirements.txt`, `packages.txt`, las carpetas `kenken/`, `models/` y `sample_images/`).
4. Haz clic en **Commit changes to main**.

---

### Paso 3: Monitorear el Build y Probar la App
1. Hugging Face comenzará a construir automáticamente el contenedor:
   * Instala los paquetes de `packages.txt`.
   * Instala las librerías de `requirements.txt`.
   * Lanza `python app.py`.
2. Puedes ver los logs en vivo haciendo clic en el botón **Logs** o **Building**.
3. En menos de 2 minutos, el estado cambiará a **Running** y verás la aplicación funcionando en vivo.
4. **¡Listo!** Puedes compartir el enlace con cualquier persona o abrirlo desde el navegador de tu celular para escanear tableros impresos con tu cámara.

---

## 4. Cómo Probar el Prototipo Localmente Antes de Subir

Puedes probar la aplicación en tu máquina local antes de subirla:

```bash
# Desde la carpeta del proyecto:
python prototipo_interactivo/app.py
```

Abre tu navegador en `http://localhost:7860`.

### 💡 Probar la Cámara del Celular con un Enlace Temporal HTTPS (`--share`)
Si deseas probar la cámara física de tu smartphone en este momento sin esperar a hacer el despliegue en Hugging Face:

```bash
python prototipo_interactivo/app.py --share
```

Gradio generará un túnel público temporal seguro:
```text
Running on local URL:  http://localhost:7860
Running on public URL: https://d19a2c4e.gradio.live
```

Copia esa URL `https://....gradio.live` y ábrela en el navegador de tu celular. Como cuenta con **HTTPS**, tu teléfono te solicitará permiso para acceder a la cámara y podrás escanear acertijos directamente.

---

## 5. Preguntas Frecuentes y Solución de Problemas

### ¿Por qué es necesario `packages.txt`?
En servidores Linux Debian/Ubuntu (como los contenedores de Hugging Face), OpenCV necesita librerías gráficas básicas del sistema como `libgl1` y `libglib2.0-0`. Al incluir `packages.txt`, Hugging Face las instala automáticamente antes de instalar los paquetes de Python.

### ¿Cómo actualizar la aplicación después del primer despliegue?
Simplemente realiza tus cambios en los archivos locales dentro de `prototipo_interactivo/` y ejecuta:
```bash
git add .
git commit -m "feat: actualizar interfaz de visualización"
git push space main
```
Hugging Face detectará el commit y reconstruirá el Space en cuestión de segundos.
