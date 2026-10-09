# KenKen Lab

Interfaz rápida para el proyecto Vision-CP. Hay dos entornos:

| Función | Localhost | GitHub Pages / servidor estático |
|---|---|---|
| Catálogo de 21 imágenes, con cuatro vistas | Sí | Sí, precalculado |
| Subir y resolver fotos nuevas | Sí, con el Python original | Requiere abrir localhost |
| Comparar A, B y C, y activar sumas redundantes | Sí | Solo resultado automático precalculado |
| Importar, editar, generar y resolver JSON | Sí, navegador | Sí, navegador |
| Descargar resultados y consultar el código | Sí | Sí |
| Ejecutar toda la suite Python desde la interfaz | Sí | GitHub Actions |
| Probar el solver del navegador | Sí | Sí |

## Iniciar en Windows

Desde la carpeta TP:

```powershell
.\iniciar-web.ps1
```

Abre http://127.0.0.1:8000. El script utiliza `.venv`; instala dependencias solo si faltan. Ctrl+C detiene el servicio que el script inicia. `-Port 8001` permite usar otro puerto y `-NoBrowser` evita abrir una ventana.

Si PowerShell bloquea la ejecución de scripts, puedes ejecutar directamente:

```powershell
.\.venv\Scripts\python.exe web/build.py
.\.venv\Scripts\python.exe -m uvicorn web.server:app --host 127.0.0.1 --port 8000
```

En un entorno nuevo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r web/requirements.txt "gradio>=5,<7"
```

## Qué se ejecuta

Las fotos nuevas pasan por `kenken.pipeline.solve_image`: OpenCV, GlyphCNN y OR-Tools CP-SAT. No se adaptaron esos modelos a WebAssembly. El servicio solo escucha en 127.0.0.1; ninguna foto se envía a un proveedor externo. Acepta hasta 12 MB y 24 megapíxeles. El tiempo de búsqueda CP se limita a 10 segundos por intento; el modo automático puede hacer dos intentos. Cancelar una petición detiene la espera de la interfaz; una tarea Python ya iniciada termina su cómputo y puede dejar su resultado en caché.

Los resultados locales se guardan en `scratch/web_cache`, con una clave que incluye imagen, método, vista, redundancia, código del motor y pesos del modelo. Cambiar el código Python o los pesos exige reiniciar el servidor para actualizar su versión de caché.

El solver JSON usa un Web Worker con propagación de tablas, filas/columnas AllDifferent y búsqueda MRV, y verifica la solución por separado. Es una implementación para navegador, no una ejecución de CP-SAT. Devuelve FEASIBLE para la primera solución válida, INFEASIBLE cuando agotó la búsqueda y UNKNOWN cuando alcanzó el límite. No realiza optimización MAP de candidatos OCR en JSON. El editor opera con las jaulas explícitas. El generador produce una instancia válida, pero no garantiza solución única.

La fidelidad de lectura mide etiquetas no corregidas, no exactitud frente al tablero real. Los resultados con ajustes incluyen un aviso para revisar la foto.

## Publicar la versión estática

El workflow `.github/workflows/pages.yml` instala Torch para CPU, construye la caché con el pipeline original, ejecuta todas las pruebas Python y del navegador y publica `web/public` en GitHub Pages. No instala un backend en la web publicada. El frontend no usa npm ni un framework en producción.

1. En GitHub, abre **Settings → Pages → Build and deployment → Source → GitHub Actions**.
2. Sube los cambios a `main`; el workflow se dispara automáticamente. También puedes usar **Actions → Test and deploy KenKen Lab → Run workflow**.
3. La URL esperada del repositorio es https://joanixx.github.io/kenken-vision-cp-solver/ . Solo está publicada cuando el job `deploy` finaliza correctamente.

Las fotos nuevas siguen requiriendo localhost. La web publicada explica esta limitación y permite probar los ejemplos y los JSON sin servidor.

Para generar o actualizar la caché de los 21 casos manualmente:

```powershell
.\.venv\Scripts\python.exe web/build.py --precompute
```

Los archivos generados están ignorados por Git. Las imágenes de salida se cargan bajo demanda. El service worker guarda la interfaz y los recursos visitados; la navegación ya usada sigue disponible sin conexión. Cada build cambia la versión de caché. La resolución de fotos nuevas requiere que el servicio local esté activo.

## Pruebas

En la pestaña **Pruebas** puedes ejecutar la suite Python completa y 14 comprobaciones del solver del navegador. También desde terminal:

```powershell
.\.venv\Scripts\python.exe -m pytest
node --test web/tests/solver.test.mjs
```

El código fuente se consulta en la pestaña **Código**. Para publicar cambios en Python, reconstruye la web para actualizar esta copia de las fuentes.

Documentación de despliegue: [workflows personalizados de GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
