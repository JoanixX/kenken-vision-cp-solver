"""Prototipo Interactivo para Hugging Face Spaces: KenKen Vision-CP Solver.

Permite:
  1. Escanear KenKen con la cámara en vivo (webcam / teléfono móvil) o subir una imagen.
  2. Ejecutar el pipeline de visión + CP-SAT con fallback automático a inferencia conjunta.
  3. Visualizar la solución proyectada, la grilla limpia y las métricas de resolución.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Asegurar que el módulo kenken se pueda importar desde este directorio
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import gradio as gr
import numpy as np

try:
    import spaces
    gpu_decorator = spaces.GPU
except Exception:
    def gpu_decorator(fn):
        return fn

from kenken.pipeline import solve_image


@gpu_decorator
def resolver_kenken(image: np.ndarray | None, method: str = "auto", render_mode: str = "composite"):
    """Callback para procesar la imagen capturada o subida en Gradio."""
    if image is None:
        return None, "⚠️ **Por favor captura una fotografía con la cámara o selecciona una imagen de los ejemplos.**"

    # Gradio entrega la imagen en RGB; convertimos a BGR para OpenCV
    if image.ndim == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image

    try:
        resultado = solve_image(img_bgr, method=method, render_mode=render_mode)

        if not resultado.solved or resultado.grid is None:
            return (
                None,
                f"❌ **No se pudo encontrar una solución válida.**\n\n"
                f"- **Estado reportado por el solver:** `{resultado.status}`\n"
                f"- **Sugerencia:** Asegúrate de que las líneas del tablero y las etiquetas aritméticas "
                f"sean claramente visibles y estén bien iluminadas.",
            )

        # Generar imagen de la solución según el modo seleccionado
        sol_bgr = resultado.render_solution(mode=render_mode)
        sol_rgb = cv2.cvtColor(sol_bgr, cv2.COLOR_BGR2RGB)

        n = resultado.instance.n
        grid_str = "\n".join(" ".join(f"{val:2d}" for val in row) for row in resultado.grid)

        # Extraer detalle de correcciones realizadas por Inferencia Conjunta
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
            conf_badge = "🟢 **Fiabilidad Muy Alta (100% Coincidencia Top-1):** Todas las jaulas leídas en la percepción visual formaron directamente un Cuadrado Latino válido sin requerir ajustes."
            fallback_msg = "⚡ **Lectura directa top-1 consistente**"
        elif conf_level == "HIGH" or not divergent:
            conf_badge = (
                f"🟢 **Fiabilidad Alta (Rescate Neuro-Simbólico):** Se ajustaron **{num_changed} jaula(s)** mediante restricciones lógicas "
                f"(**{fidelity}% de fidelidad visual**). El solver dedujo las lecturas correctas del Top-$k$ con alta certidumbre matemática."
            )
            fallback_msg = f"✅ **Activada ({num_changed} jaula(s) corregida(s), fidelidad {fidelity}%)**"
        else:
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

        metricas_md = f"""
### 🧩 KenKen {n}×{n} Resuelto Exitosamente

| Métrica | Valor |
|---|---|
| **Estado del Solver** | `{resultado.status}` |
| **Fidelidad Visual** | `{fidelity}%` ({resultado.confidence_level}) |
| **Tiempo de CPU** | `{resultado.solve_result.wall_time * 1000:.2f} ms` |
| **Ramas Exploradas** | `{resultado.solve_result.branches}` |
| **Conflictos Resueltos** | `{resultado.solve_result.conflicts}` |
| **Jaulas Detectadas** | `{len(resultado.instance.cages)}` |
| **Inferencia Conjunta (MAP)** | {fallback_msg} |

{diagnostico_section}

#### Matriz Solución:
```text
{grid_str}
```
"""
        return sol_rgb, metricas_md

    except Exception as e:
        return (
            None,
            f"⚠️ **Error en el procesamiento de la imagen:** `{type(e).__name__}: {str(e)}`\n\n"
            f"Verifica que el encuadre contenga un tablero KenKen cuadrangular identificable.",
        )


def build_app() -> gr.Blocks:
    """Construye la interfaz gráfica interactiva de Gradio."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    )

    sample_dir = ROOT_DIR / "sample_images"
    category_groups = [
        {
            "name": "🟢 1. Lectura Directa",
            "desc": "Imágenes digitales nítidas donde la percepción visual (Top-1) acierta al 100% y CP-SAT resuelve en <0.2 s sin requerir fallback.",
            "examples": [
                ["tipo1_directo_3x3.png", "auto", "composite", "Directo 3×3 (Nítido)"],
                ["tipo1_directo_4x4.png", "auto", "composite", "Directo 4×4 (Nítido)"],
                ["tipo1_directo_5x5.jpg", "auto", "composite", "Directo 5×5 (Nítido)"],
            ],
        },
        {
            "name": "⚡ 2. Rescate MAP (Ambigüedad)",
            "desc": "Ambigüedad leve en operadores (+ vs /) donde la inferencia conjunta (Variante C) deduce la operación correcta por consistencia lógica.",
            "examples": [
                ["tipo2_map_3x3.jpg", "auto", "composite", "MAP 3×3 (1 corrección)"],
                ["tipo2_map_4x4.jpg", "auto", "composite", "MAP 4×4 (1 corrección)"],
                ["tipo2_map_5x5.jpg", "auto", "composite", "MAP 5×5 (2 correcciones)"],
            ],
        },
        {
            "name": "📐 3. Perspectiva y Rotación",
            "desc": "Fotografías con perspectiva angular, inclinación y distorsión geométrica, rectificadas por homografía en OpenCV.",
            "examples": [
                ["tipo3_perspectiva_4x4.jpg", "auto", "composite", "Perspectiva 4×4 (Ángulo)"],
                ["tipo3_perspectiva_5x5.jpg", "auto", "composite", "Perspectiva 5×5 (Inclinación)"],
                ["tipo3_perspectiva_6x6.jpg", "auto", "composite", "Perspectiva 6×6 (Rotación)"],
            ],
        },
        {
            "name": "🏆 4. Gran Escala y Complejidad",
            "desc": "Tableros de alta dimensión y densidad combinatoria (6×6 a 9×9) con multiplicaciones grandes (e.g. 240*), demostrando la escalabilidad de CP-SAT.",
            "examples": [
                ["tipo4_gran_escala_6x6.jpg", "auto", "composite", "Gran Escala 6×6"],
                ["tipo4_gran_escala_9x9_foto.jpg", "auto", "composite", "Foto Compleja 9×9"],
                ["tipo4_gran_escala_9x9_limpio.png", "auto", "composite", "Experto Oficial 9×9"],
            ],
        },
        {
            "name": "⚡ 5. Rescate Avanzado en Estrés",
            "desc": "Casos de estrés extremo donde MAP rescata simultáneamente 1, 2 y hasta 3 jaulas borrosas con 100% de coincidencia exacta con el Ground Truth.",
            "examples": [
                ["tipo5_rescate_estres_1cambio.jpg", "auto", "composite", "Rescate 1 Jaula (93.3% fid.)"],
                ["tipo5_rescate_estres_2cambios.jpg", "auto", "composite", "Rescate 2 Jaulas (90.9% fid.)"],
                ["tipo5_rescate_estres_3cambios.jpg", "auto", "composite", "Rescate 3 Jaulas (82.4% fid.)"],
            ],
        },
        {
            "name": "🟡 6. Aviso (Múltiples Ajustes)",
            "desc": "Imágenes con degradación severa donde el solver encuentra solución pero emite un aviso amarillo preventivo recomendando verificar el impreso.",
            "examples": [
                ["tipo6_aviso_ajustes_7jaulas.jpg", "auto", "composite", "Aviso 7 Ajustes (63.2% fid.)"],
                ["tipo6_aviso_ajustes_8jaulas.jpg", "auto", "composite", "Aviso 8 Ajustes (52.9% fid.)"],
                ["tipo6_aviso_ajustes_14jaulas.jpg", "auto", "composite", "Aviso 14 Ajustes (46.2% fid.)"],
            ],
        },
        {
            "name": "❌ 7. Infactible (Contradicciones)",
            "desc": "Tableros con contradicciones matemáticas insolubles donde el solver diagnostica formalmente la condición INFEASIBLE.",
            "examples": [
                ["tipo7_infactible_muestra1.jpg", "auto", "composite", "Infactible Muestra 1"],
                ["tipo7_infactible_muestra2.jpg", "auto", "composite", "Infactible Muestra 2"],
                ["tipo7_infactible_muestra3.jpg", "auto", "composite", "Infactible Muestra 3"],
            ],
        },
    ]

    with gr.Blocks(title="KenKen Vision-CP Solver") as demo:
        gr.Markdown(
            """
            # 🧩 KenKen Vision-CP Solver
            ### Sistema Híbrido Neuro-Simbólico de Visión por Computador y Programación por Restricciones
            Apunta tu **cámara web o de teléfono móvil** a un acertijo KenKen o selecciona una imagen para resolverlo en tiempo real.
            Equipado con **GlyphCNN Fine-Tuned (Focal Loss + Hard Example Mining)** y solver **Google OR-Tools CP-SAT**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 1. Captura o Selección de Imagen")
                image_input = gr.Image(
                    label="Escanea con la Cámara o Sube una Foto",
                    sources=["webcam", "upload", "clipboard"],
                    type="numpy",
                )

                with gr.Accordion("⚙️ Opciones del Solver y Visualización", open=False):
                    method_input = gr.Radio(
                        choices=["auto", "table", "arithmetic", "joint"],
                        value="auto",
                        label="Estrategia del Solver CP",
                        info="'auto' intenta lectura directa y activa inferencia conjunta si detecta error.",
                    )
                    render_input = gr.Radio(
                        choices=["composite", "clean", "original", "rectified"],
                        value="composite",
                        label="Modo de Visualización",
                        info="'composite' muestra comparativa lado a lado; 'clean' muestra el tablero vectorial.",
                    )

                btn_solve = gr.Button("⚡ Resolver KenKen", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 🎯 2. Resultado y Métricas")
                image_output = gr.Image(label="Solución Visual Resuelta", type="numpy")
                metrics_output = gr.Markdown("*(Los resultados y estadísticas aparecerán aquí tras la ejecución)*")

        gr.Markdown("---")
        gr.Markdown("### 📂 Catálogo Demostrativo por Categorías (3 Imágenes por Tipo de Muestra)")
        gr.Markdown(
            "💡 *Explora las vistas previas organizadas por categoría técnica. "
            "Haz clic directamente sobre la imagen o en **⚡ Resolver** para probarla al instante, "
            "o en **📥 Cargar** para inspeccionarla en el panel de entrada.*"
        )
        with gr.Tabs():
            for group in category_groups:
                with gr.Tab(group["name"]):
                    gr.Markdown(f"*{group['desc']}*")
                    with gr.Row():
                        for fname, method, mode, lbl in group["examples"]:
                            fpath = sample_dir / fname
                            if fpath.exists():
                                with gr.Column(scale=1):
                                    preview_img = gr.Image(
                                        value=str(fpath),
                                        label=lbl,
                                        interactive=False,
                                        height=210,
                                        show_label=True,
                                    )
                                    with gr.Row():
                                        btn_load_only = gr.Button("📥 Cargar", variant="secondary", size="sm")
                                        btn_solve_now = gr.Button("⚡ Resolver", variant="primary", size="sm")

                                    def _make_sample_handlers(img_p=str(fpath), m=method, r=mode):
                                        def _load_handler(*_args):
                                            img_b = cv2.imread(img_p)
                                            if img_b is None:
                                                return None, m, r
                                            img_r = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
                                            return img_r, m, r

                                        def _solve_handler(*_args):
                                            img_b = cv2.imread(img_p)
                                            if img_b is None:
                                                return None, m, r, None, "⚠️ Error: No se pudo cargar el archivo de imagen."
                                            img_r = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
                                            sol_r, met_md = resolver_kenken(img_r, method=m, render_mode=r)
                                            return img_r, m, r, sol_r, met_md

                                        return _load_handler, _solve_handler

                                    load_fn, solve_fn = _make_sample_handlers(str(fpath), method, mode)
                                    btn_load_only.click(
                                        fn=load_fn,
                                        inputs=[],
                                        outputs=[image_input, method_input, render_input],
                                    )
                                    btn_solve_now.click(
                                        fn=solve_fn,
                                        inputs=[],
                                        outputs=[image_input, method_input, render_input, image_output, metrics_output],
                                    )
                                    preview_img.select(
                                        fn=solve_fn,
                                        inputs=[],
                                        outputs=[image_input, method_input, render_input, image_output, metrics_output],
                                    )

        btn_solve.click(
            fn=resolver_kenken,
            inputs=[image_input, method_input, render_input],
            outputs=[image_output, metrics_output],
        )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #64748b; font-size: 0.9em;">
            Desarrollado con <b>OpenCV</b>, <b>PyTorch GlyphCNN (Fine-Tuned)</b> y <b>Google OR-Tools CP-SAT</b> | Alojado en Hugging Face Spaces
            </div>
            """
        )

    return demo


demo = build_app()


def main():
    parser = argparse.ArgumentParser(description="KenKen Solver Gradio App para Hugging Face Spaces")
    parser.add_argument("--host", default="0.0.0.0", help="Dirección host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Puerto de escucha (default: 7860)")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generar enlace público temporal HTTPS (ideal para probar en smartphone)",
    )
    args = parser.parse_args()

    print(f"\n[✓] Iniciando servidor Gradio en http://{args.host}:{args.port}")
    if args.share:
        print("[✓] Generando enlace público HTTPS para acceso móvil...")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
