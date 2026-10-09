"""Prototipo Interactivo para Streamlit y Hugging Face Spaces: KenKen Vision-CP Solver.

Sistema neuro-simbólico que combina Visión por Computador (OpenCV + GlyphCNN Fine-Tuned)
y Programación por Restricciones (Google OR-Tools CP-SAT).

Permite:
  1. Escanear KenKen con la cámara en vivo (webcam / teléfono móvil) o subir una imagen.
  2. Explorar el catálogo demostrativo con los 21 casos del TP (7 categorías técnicas).
  3. Ejecutar el pipeline de visión + CP-SAT con fallback automático a inferencia conjunta (MAP).
  4. Visualizar la solución proyectada, tableros limpios, métricas cuantitativas y diagnósticos.
"""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
from typing import Any

# Asegurar que el módulo kenken se pueda importar desde este directorio
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from kenken.pipeline import solve_image

SAMPLE_DIR = ROOT_DIR / "sample_images"

# Catálogo completo con las 21 muestras divididas en las 7 categorías del TP
CATEGORY_GROUPS: list[dict[str, Any]] = [
    {
        "id": "directo",
        "name": "🟢 1. Lectura Directa",
        "badge": "100% Precisión Visual",
        "desc": "Imágenes digitales nítidas donde la percepción visual (Top-1) acierta al 100% y CP-SAT resuelve en <0.2 s sin requerir fallback.",
        "examples": [
            {"file": "tipo1_directo_3x3.png", "title": "Directo 3×3 (Nítido)", "method": "auto", "render": "composite"},
            {"file": "tipo1_directo_4x4.png", "title": "Directo 4×4 (Nítido)", "method": "auto", "render": "composite"},
            {"file": "tipo1_directo_5x5.jpg", "title": "Directo 5×5 (Nítido)", "method": "auto", "render": "composite"},
        ],
    },
    {
        "id": "map",
        "name": "⚡ 2. Rescate MAP (Ambigüedad)",
        "badge": "Inferencia Conjunta",
        "desc": "Ambigüedad leve en operadores (+ vs /) donde la inferencia conjunta (Variante C) deduce la operación correcta por consistencia lógica.",
        "examples": [
            {"file": "tipo2_map_3x3.jpg", "title": "MAP 3×3 (1 corrección)", "method": "auto", "render": "composite"},
            {"file": "tipo2_map_4x4.jpg", "title": "MAP 4×4 (1 corrección)", "method": "auto", "render": "composite"},
            {"file": "tipo2_map_5x5.jpg", "title": "MAP 5×5 (2 correcciones)", "method": "auto", "render": "composite"},
        ],
    },
    {
        "id": "perspectiva",
        "name": "📐 3. Perspectiva y Rotación",
        "badge": "Homografía OpenCV",
        "desc": "Fotografías con perspectiva angular, inclinación y distorsión geométrica, rectificadas por homografía en OpenCV.",
        "examples": [
            {"file": "tipo3_perspectiva_4x4.jpg", "title": "Perspectiva 4×4 (Ángulo)", "method": "auto", "render": "composite"},
            {"file": "tipo3_perspectiva_5x5.jpg", "title": "Perspectiva 5×5 (Inclinación)", "method": "auto", "render": "composite"},
            {"file": "tipo3_perspectiva_6x6.jpg", "title": "Perspectiva 6×6 (Rotación)", "method": "auto", "render": "composite"},
        ],
    },
    {
        "id": "escala",
        "name": "🏆 4. Gran Escala y Complejidad",
        "badge": "Escalabilidad CP-SAT",
        "desc": "Tableros de alta dimensión y densidad combinatoria (6×6 a 9×9) con multiplicaciones grandes (ej. 240×), demostrando la escalabilidad de CP-SAT.",
        "examples": [
            {"file": "tipo4_gran_escala_6x6.jpg", "title": "Gran Escala 6×6", "method": "auto", "render": "composite"},
            {"file": "tipo4_gran_escala_9x9_foto.jpg", "title": "Foto Compleja 9×9", "method": "auto", "render": "composite"},
            {"file": "tipo4_gran_escala_9x9_limpio.png", "title": "Experto Oficial 9×9", "method": "auto", "render": "composite"},
        ],
    },
    {
        "id": "estres",
        "name": "⚡ 5. Rescate Avanzado en Estrés",
        "badge": "Estrés Extremo",
        "desc": "Casos de estrés extremo donde MAP rescata simultáneamente 1, 2 y hasta 3 jaulas borrosas con 100% de coincidencia exacta con el Ground Truth.",
        "examples": [
            {"file": "tipo5_rescate_estres_1cambio.jpg", "title": "Rescate 1 Jaula (93.3% fid.)", "method": "auto", "render": "composite"},
            {"file": "tipo5_rescate_estres_2cambios.jpg", "title": "Rescate 2 Jaulas (90.9% fid.)", "method": "auto", "render": "composite"},
            {"file": "tipo5_rescate_estres_3cambios.jpg", "title": "Rescate 3 Jaulas (82.4% fid.)", "method": "auto", "render": "composite"},
        ],
    },
    {
        "id": "aviso",
        "name": "🟡 6. Aviso (Múltiples Ajustes)",
        "badge": "Clasificación Selectiva",
        "desc": "Imágenes con degradación severa donde el solver encuentra solución pero emite un aviso preventivo recomendando verificar el impreso.",
        "examples": [
            {"file": "tipo6_aviso_ajustes_7jaulas.jpg", "title": "Aviso 7 Ajustes (63.2% fid.)", "method": "auto", "render": "composite"},
            {"file": "tipo6_aviso_ajustes_8jaulas.jpg", "title": "Aviso 8 Ajustes (52.9% fid.)", "method": "auto", "render": "composite"},
            {"file": "tipo6_aviso_ajustes_14jaulas.jpg", "title": "Aviso 14 Ajustes (46.2% fid.)", "method": "auto", "render": "composite"},
        ],
    },
    {
        "id": "infactible",
        "name": "❌ 7. Infactible (Contradicciones)",
        "badge": "Diagnóstico Formal",
        "desc": "Tableros con contradicciones matemáticas insolubles donde el solver diagnostica formalmente la condición INFEASIBLE.",
        "examples": [
            {"file": "tipo7_infactible_muestra1.jpg", "title": "Infactible Muestra 1", "method": "auto", "render": "composite"},
            {"file": "tipo7_infactible_muestra2.jpg", "title": "Infactible Muestra 2", "method": "auto", "render": "composite"},
            {"file": "tipo7_infactible_muestra3.jpg", "title": "Infactible Muestra 3", "method": "auto", "render": "composite"},
        ],
    },
]


def resolver_kenken_detallado(
    image: np.ndarray | None,
    method: str = "auto",
    render_mode: str = "composite",
) -> dict[str, Any]:
    """Ejecuta el pipeline completo y retorna un diccionario con resultados detallados."""
    if image is None:
        return {
            "solved": False,
            "sol_rgb": None,
            "status": "NO_INPUT",
            "message": "⚠️ **Por favor captura una fotografía con la cámara o selecciona una imagen de los ejemplos.**",
            "metrics_md": "⚠️ **Por favor captura una fotografía con la cámara o selecciona una imagen de los ejemplos.**",
            "corrections": [],
            "grid_str": "",
        }

    # Streamlit entrega la imagen en RGB; convertimos a BGR para OpenCV
    if image.ndim == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image

    try:
        resultado = solve_image(img_bgr, method=method, render_mode=render_mode)

        if not resultado.solved or resultado.grid is None:
            err_msg = (
                f"❌ **No se pudo encontrar una solución válida.**\n\n"
                f"- **Estado reportado por el solver:** `{resultado.status}`\n"
                f"- **Sugerencia:** Asegúrate de que las líneas del tablero y las etiquetas aritméticas "
                f"sean claramente visibles y estén bien iluminadas."
            )
            return {
                "solved": False,
                "sol_rgb": None,
                "status": resultado.status,
                "message": err_msg,
                "metrics_md": err_msg,
                "corrections": [],
                "grid_str": "",
                "raw_result": resultado,
            }

        # Generar imagen de la solución según el modo seleccionado
        sol_bgr = resultado.render_solution(mode=render_mode)
        sol_rgb = cv2.cvtColor(sol_bgr, cv2.COLOR_BGR2RGB)

        n = resultado.instance.n
        grid_str = "\n".join(" ".join(f"{val:2d}" for val in row) for row in resultado.grid)

        # Extraer detalle de correcciones realizadas por Inferencia Conjunta (MAP)
        corrections = []
        if resultado.fallback_used and resultado.solve_result.chosen_candidates:
            for c_idx, chosen in resultado.solve_result.chosen_candidates.items():
                top1 = resultado.instance.candidates.get(c_idx, [{}])[0]
                t1_target = top1.get("target")
                t1_op = top1.get("op", "")
                ch_target = chosen.get("target")
                ch_op = chosen.get("op", "")
                if ch_target != t1_target or ch_op != t1_op:
                    cells_str = ", ".join(f"({r},{c})" for r, c in resultado.instance.cages[c_idx].cells)
                    corrections.append(
                        f"- **Jaula {c_idx}** [{cells_str}]: Top-1 visual `{t1_target}{t1_op}` ➔ Ajustado a `{ch_target}{ch_op}`"
                    )

        fidelity = getattr(resultado, "fidelity", 100.0)
        num_changed = getattr(resultado, "num_changed", len(corrections))
        conf_level = getattr(resultado, "confidence_level", "HIGH")
        divergent = getattr(resultado, "divergent", False)

        if not resultado.fallback_used or num_changed == 0:
            diag_category = "success"
            conf_badge = "🟢 **Fiabilidad Muy Alta (100% Coincidencia Top-1):** Todas las jaulas leídas en la percepción visual formaron directamente un Cuadrado Latino válido sin requerir ajustes."
            fallback_msg = "⚡ **Lectura directa top-1 consistente**"
        elif conf_level == "HIGH" or not divergent:
            diag_category = "success"
            conf_badge = (
                f"🟢 **Fiabilidad Alta (Rescate Neuro-Simbólico):** Se ajustaron **{num_changed} jaula(s)** mediante restricciones lógicas "
                f"(**{fidelity}% de fidelidad visual**). El solver dedujo las lecturas correctas del Top-$k$ con alta certidumbre matemática."
            )
            fallback_msg = f"✅ **Activada ({num_changed} jaula(s) corregida(s), fidelidad {fidelity}%)**"
        else:
            diag_category = "warning"
            conf_badge = (
                f"🟡 **Aviso de Verificación (Múltiples Ajustes):** Se requirieron **{num_changed} correcciones** de jaulas (**{fidelity}% de fidelidad visual**). "
                f"La solución matemática encontrada satisface formalmente todas las reglas de KenKen; te recomendamos revisar el desglose de jaulas abajo para confirmar que coincidan con tu impreso si alguna etiqueta visual fue muy borrosa."
            )
            fallback_msg = f"⚠️ **Activada con múltiples ajustes ({num_changed} jaulas, fidelidad {fidelity}%)**"

        corr_details = "\n".join(corrections) if corrections else "*(Sin cambios sobre top-1)*"
        diagnostico_section = f"""
#### 🔍 Diagnóstico de Fiabilidad y Correcciones:
{conf_badge}

{corr_details}
"""

        wall_time_ms = resultado.solve_result.wall_time * 1000.0
        branches = resultado.solve_result.branches
        conflicts = resultado.solve_result.conflicts
        num_cages = len(resultado.instance.cages)

        metricas_md = f"""
### 🧩 KenKen {n}×{n} Resuelto Exitosamente

| Métrica | Valor |
|---|---|
| **Estado del Solver** | `{resultado.status}` |
| **Fidelidad Visual** | `{fidelity}%` ({conf_level}) |
| **Tiempo de CPU** | `{wall_time_ms:.2f} ms` |
| **Ramas Exploradas** | `{branches}` |
| **Conflictos Resueltos** | `{conflicts}` |
| **Jaulas Detectadas** | `{num_cages}` |
| **Inferencia Conjunta (MAP)** | {fallback_msg} |

{diagnostico_section}

#### Matriz Solución:
```text
{grid_str}
```
"""
        return {
            "solved": True,
            "sol_rgb": sol_rgb,
            "status": resultado.status,
            "message": "KenKen resuelto exitosamente.",
            "metrics_md": metricas_md,
            "n": n,
            "fidelity": fidelity,
            "confidence_level": conf_level,
            "wall_time_ms": wall_time_ms,
            "branches": branches,
            "conflicts": conflicts,
            "num_cages": num_cages,
            "fallback_used": resultado.fallback_used,
            "fallback_msg": fallback_msg,
            "diag_category": diag_category,
            "conf_badge": conf_badge,
            "corrections": corrections,
            "grid": resultado.grid,
            "grid_str": grid_str,
            "raw_result": resultado,
        }

    except Exception as e:
        err_msg = (
            f"⚠️ **Error en el procesamiento de la imagen:** `{type(e).__name__}: {str(e)}`\n\n"
            f"Verifica que el encuadre contenga un tablero KenKen cuadrangular identificable."
        )
        return {
            "solved": False,
            "sol_rgb": None,
            "status": "ERROR",
            "message": err_msg,
            "metrics_md": err_msg,
            "corrections": [],
            "grid_str": "",
        }


def resolver_kenken(
    image: np.ndarray | None,
    method: str = "auto",
    render_mode: str = "composite",
) -> tuple[np.ndarray | None, str]:
    """Función de compatibilidad directa para pruebas automatizadas y API.

    Retorna una tupla: (imagen_solucion_rgb, texto_metricas_md).
    """
    res = resolver_kenken_detallado(image, method=method, render_mode=render_mode)
    return res["sol_rgb"], res["metrics_md"]


def build_app() -> dict[str, Any]:
    """Retorna los metadatos de la aplicación para suites de pruebas."""
    return {
        "title": "KenKen Vision-CP Solver",
        "framework": "Streamlit",
        "categories": [c["name"] for c in CATEGORY_GROUPS],
        "total_cases": sum(len(c["examples"]) for c in CATEGORY_GROUPS),
    }


def _img_to_bytes(img_rgb: np.ndarray) -> bytes:
    """Convierte un array RGB a bytes PNG para descarga."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def run_streamlit_app():
    """Renderiza la interfaz gráfica en Streamlit."""
    st.set_page_config(
        page_title="KenKen Vision-CP Solver",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inyección de estilos CSS para un aspecto profesional y limpio
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .tech-badge {
            display: inline-block;
            background: #e0f2fe;
            color: #0369a1;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inicializar estado de sesión
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_image_source" not in st.session_state:
        st.session_state.current_image_source = None
    if "last_solved_result" not in st.session_state:
        st.session_state.last_solved_result = None

    # ========================== SIDEBAR ==========================
    with st.sidebar:
        st.title("🧩 KenKen Solver")
        st.markdown(
            """
            <span class="tech-badge">OpenCV</span>
            <span class="tech-badge">PyTorch CNN</span>
            <span class="tech-badge">OR-Tools CP-SAT</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.subheader("⚙️ Parámetros del Solver")
        solver_method = st.radio(
            "Estrategia de Modelado CP",
            options=["auto", "table", "arithmetic", "joint"],
            format_func=lambda x: {
                "auto": "auto (Directo + Fallback MAP)",
                "joint": "joint (Inferencia Conjunta MAP)",
                "arithmetic": "arithmetic (Restricciones Aritméticas)",
                "table": "table (Restricciones Extensionales)",
            }[x],
            help="'auto' intenta lectura directa rápida y activa MAP si detecta incoherencia.",
        )

        render_mode = st.radio(
            "Modo de Visualización",
            options=["composite", "clean", "original", "rectified"],
            format_func=lambda x: {
                "composite": "composite (Comparativa Lado a Lado)",
                "clean": "clean (Tablero Vectorial Nítido)",
                "original": "original (Proyección Perspectiva)",
                "rectified": "rectified (Tablero Rectificado)",
            }[x],
            help="Selecciona cómo representar la solución final.",
        )

        st.markdown("---")
        st.subheader("📖 Información Académica")
        st.caption(
            "**Curso:** CC58 - Tópicos en CC (UPC)\n\n"
            "**Trabajo Parcial (TP):** Integración Neuro-Simbólica de Visión Computacional "
            "y Programación por Restricciones.\n\n"
            "**Solver:** Google OR-Tools CP-SAT con Lazy Clause Generation."
        )

        if st.button("🔄 Limpiar / Reiniciar Todo", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.current_image_source = None
            st.session_state.last_solved_result = None
            st.rerun()

    # ========================== CONTENIDO PRINCIPAL ==========================
    st.markdown('<div class="main-header">🧩 KenKen Vision-CP Solver</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Sistema Híbrido Neuro-Simbólico: Percepción con OpenCV + GlyphCNN y Razonamiento Lógico con OR-Tools CP-SAT</div>',
        unsafe_allow_html=True,
    )

    # Pestañas para los 3 métodos de entrada
    tab_cam, tab_up, tab_cat = st.tabs([
        "📸 Cámara en Vivo (Webcam / Móvil)",
        "📁 Subir Imagen",
        "📂 Catálogo Demostrativo (21 Casos del TP)",
    ])

    # 1. Cámara en Vivo
    with tab_cam:
        st.info("💡 **Acceso Móvil / Web:** Apunta la cámara directamente hacia el tablero impreso asegurando buena iluminación y enfoque de las líneas.")
        cam_photo = st.camera_input("Capturar foto del tablero KenKen")
        if cam_photo is not None:
            pil_cam = Image.open(cam_photo).convert("RGB")
            st.session_state.current_image = np.array(pil_cam)
            st.session_state.current_image_source = "Cámara en Vivo"

    # 2. Subir Archivo
    with tab_up:
        uploaded_file = st.file_uploader(
            "Selecciona o arrastra una imagen de KenKen (.png, .jpg, .jpeg)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            pil_up = Image.open(uploaded_file).convert("RGB")
            st.session_state.current_image = np.array(pil_up)
            st.session_state.current_image_source = f"Archivo: {uploaded_file.name}"

    # 3. Catálogo Demostrativo por Categorías
    with tab_cat:
        st.markdown(
            "💡 *Explora los 21 casos del dataset experimental organizados en las 7 categorías del informe técnico. "
            "Haz clic en **⚡ Probar este Caso** para cargarlo y resolverlo de inmediato.*"
        )
        cat_tabs = st.tabs([cat["name"] for cat in CATEGORY_GROUPS])
        for idx_c, cat in enumerate(CATEGORY_GROUPS):
            with cat_tabs[idx_c]:
                st.caption(f"**Descripción:** {cat['desc']}")
                cols = st.columns(len(cat["examples"]))
                for col_idx, ex in enumerate(cat["examples"]):
                    with cols[col_idx]:
                        img_path = SAMPLE_DIR / ex["file"]
                        if img_path.exists():
                            st.image(str(img_path), caption=ex["title"], use_container_width=True)
                            if st.button(f"⚡ Probar: {ex['title']}", key=f"btn_cat_{cat['id']}_{col_idx}", use_container_width=True):
                                img_b = cv2.imread(str(img_path))
                                if img_b is not None:
                                    img_r = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
                                    st.session_state.current_image = img_r
                                    st.session_state.current_image_source = f"Catálogo: {ex['title']}"
                                    # Resolver de inmediato
                                    with st.spinner("Ejecutando pipeline de visión y CP-SAT..."):
                                        res = resolver_kenken_detallado(img_r, method=ex.get("method", solver_method), render_mode=ex.get("render", render_mode))
                                        st.session_state.last_solved_result = res
                                    st.rerun()
                        else:
                            st.warning(f"Archivo no encontrado: {ex['file']}")

    st.markdown("---")

    # ========================== PANEL DE PROCESAMIENTO Y RESULTADOS ==========================
    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("📷 1. Tablero Seleccionado")
        if st.session_state.current_image is not None:
            st.image(
                st.session_state.current_image,
                caption=f"Origen: {st.session_state.current_image_source or 'Personalizado'}",
                use_container_width=True,
            )
            btn_run = st.button("⚡ Resolver KenKen", type="primary", use_container_width=True)
            if btn_run:
                with st.spinner("Ejecutando pipeline de visión por computador y solver CP-SAT..."):
                    res = resolver_kenken_detallado(
                        st.session_state.current_image,
                        method=solver_method,
                        render_mode=render_mode,
                    )
                    st.session_state.last_solved_result = res
                st.rerun()
        else:
            st.info("👆 Selecciona una muestra del catálogo, sube un archivo o toma una fotografía con tu cámara.")

    with col_output:
        st.subheader("🎯 2. Resultado y Métricas")
        res = st.session_state.last_solved_result

        if res is not None:
            if res.get("solved", False) and res.get("sol_rgb") is not None:
                st.image(res["sol_rgb"], caption=f"Solución KenKen {res['n']}×{res['n']} ({render_mode})", use_container_width=True)

                # Botón para descargar la solución
                img_bytes = _img_to_bytes(res["sol_rgb"])
                st.download_button(
                    label="💾 Descargar Imagen de la Solución",
                    data=img_bytes,
                    file_name=f"kenken_{res['n']}x{res['n']}_solucion.png",
                    mime="image/png",
                    use_container_width=True,
                )

                # Métricas cuantitativas
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("⏱️ CPU Solver", f"{res['wall_time_ms']:.1f} ms")
                with m2:
                    st.metric("🎯 Fidelidad Visual", f"{res['fidelity']:.1f}%")
                with m3:
                    st.metric("🌳 Ramas CP-SAT", str(res["branches"]))
                with m4:
                    st.metric("⚡ Conflictos", str(res["conflicts"]))

                # Banner de diagnóstico
                if res.get("diag_category") == "success":
                    st.success(res["conf_badge"])
                else:
                    st.warning(res["conf_badge"])

                # Desglose de correcciones si hubo rescate MAP
                if res.get("corrections"):
                    with st.expander(f"🔍 Ver Correcciones por Inferencia Conjunta ({len(res['corrections'])})", expanded=True):
                        for c in res["corrections"]:
                            st.markdown(c)

                # Matriz solución
                with st.expander("🔢 Ver Matriz Solución Formal (Texto)", expanded=False):
                    st.code(res["grid_str"], language="text")

            else:
                st.error(res.get("message", "No se encontró solución."))
                if res.get("status") == "INFEASIBLE":
                    st.info("ℹ️ **Diagnóstico:** El modelo determinó formalmente que el conjunto de jaulas y operadores contiene contradicciones lógicas insolubles.")
        else:
            st.caption("*(Los resultados visuales y las métricas de rendimiento aparecerán aquí tras la resolución)*")

    # Pie de página
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
        Desarrollado para <b>CC58 Tópicos en Ciencia de la Computación</b> | Arquitectura Neuro-Simbólica OpenCV + PyTorch GlyphCNN + Google OR-Tools CP-SAT
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Punto de entrada de ejecución."""
    # Verificar si se está ejecutando bajo streamlit
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit = get_script_run_ctx() is not None
    except Exception:
        is_streamlit = False

    if is_streamlit:
        run_streamlit_app()
    else:
        # Si fue ejecutado directamente con `python app.py`, lanzarlo con `streamlit run`
        print("\n[i] Iniciando servidor Streamlit para KenKen Vision-CP Solver...")
        curr_file = Path(__file__).resolve()
        cmd = [sys.executable, "-m", "streamlit", "run", str(curr_file)] + sys.argv[1:]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n[✓] Servidor Streamlit finalizado.")


if __name__ == "__main__":
    main()
