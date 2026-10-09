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

from kenken.pipeline import solve_image


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
            if getattr(resultado, "divergent", False):
                sol_bgr = resultado.render_solution(mode=render_mode)
                sol_rgb = cv2.cvtColor(sol_bgr, cv2.COLOR_BGR2RGB)
                num_c = resultado.num_changed
                max_a = resultado.max_allowed_changes
                return (
                    sol_rgb,
                    f"### ⚠️ Advertencia: Lectura Visual No Fidedigna (Abstención Confiable)\n\n"
                    f"- **Estado Oficial:** `UNRELIABLE_DETECTION`\n"
                    f"- **Inconsistencias:** Se requirieron **{num_c} correcciones** de jaulas (límite de seguridad: **{max_a}**).\n\n"
                    f"Para no entregarte un acertijo inventado que no coincide con el impreso original, "
                    f"**el sistema se abstiene de proclamar éxito**.\n\n"
                    f"💡 **Sugerencia:** Toma una fotografía con encuadre frontal, iluminación uniforme y sin sombras sobre las etiquetas.\n\n"
                    f"*(Nota: La imagen adjunta muestra la solución matemática hipotética encontrada por el solver, pero difiere del impreso original).* ",
                )
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
                        f"- **Jaula {c_idx}** [{cells_str}]: Top-1 visual `{t1_target}{t1_op}` ➔ Corregido a `{ch_target}{ch_op}`"
                    )

        if resultado.fallback_used:
            num_corr = len(corrections)
            if num_corr <= 2:
                conf_badge = f"🟢 **Confianza Alta:** Se corrigieron {num_corr} lectura(s) ambigua(s)."
            else:
                conf_badge = (
                    f"⚠️ **Alerta de Posible Divergencia:** Se corrigieron {num_corr} jaulas respecto a la imagen original. "
                    f"Si la etiqueta real no estaba en los candidatos del OCR, la solución podría ser una variante válida diferente al impreso."
                )
            corr_details = "\n".join(corrections) if corrections else "*(Sin cambios sobre top-1)*"
            diagnostico_section = f"""
#### 🔍 Diagnóstico de Correcciones (MAP):
{conf_badge}

{corr_details}
"""
            fallback_msg = f"✅ **Activada ({num_corr} jaula(s) modificada(s))**"
        else:
            diagnostico_section = """
#### 🔍 Diagnóstico de Lectura:
⚡ **100% Consistente:** Todas las jaulas leídas en top-1 formaron directamente un Cuadrado Latino válido sin requerir modificaciones.
"""
            fallback_msg = "⚡ **Lectura directa top-1 consistente**"

        metricas_md = f"""
### 🧩 KenKen {n}×{n} Resuelto Exitosamente

| Métrica | Valor |
|---|---|
| **Estado del Solver** | `{resultado.status}` |
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
    cases_def = [
        (
            "caso1_lectura_directa_4x4.png",
            "auto",
            "composite",
            "🟢 Caso 1: Lectura Directa (4x4)",
        ),
        (
            "caso2_rescate_map_4x4.jpg",
            "auto",
            "composite",
            "⚡ Caso 2: Rescate MAP (4x4)",
        ),
        (
            "caso3_perspectiva_rotacion_5x5.jpg",
            "auto",
            "composite",
            "📐 Caso 3: Perspectiva y Rotación (5x5)",
        ),
        (
            "caso4_alta_dificultad_6x6.jpg",
            "auto",
            "composite",
            "🧩 Caso 4: Alta Dificultad (6x6)",
        ),
        (
            "caso5_gran_escala_9x9.png",
            "auto",
            "composite",
            "🏆 Caso 5: Gran Escala (9x9)",
        ),
        (
            "caso6_abstencion_confiable_divergente.jpg",
            "auto",
            "composite",
            "⚠️ Caso 6: Abstención Honesta",
        ),
        (
            "caso7_infactible_contradictorio.jpg",
            "auto",
            "composite",
            "❌ Caso 7: Tablero Infactible",
        ),
    ]

    valid_examples = []
    example_labels = []
    for fname, method, mode, label in cases_def:
        fpath = sample_dir / fname
        if fpath.exists():
            valid_examples.append([str(fpath), method, mode])
            example_labels.append(label)

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

        if valid_examples:
            gr.Markdown("---")
            gr.Markdown("### 📂 Casos de Prueba Demostrativos")
            gr.Markdown(
                """
                Selecciona cualquiera de las muestras por defecto para evaluar el comportamiento del sistema ante distintos escenarios reales:

                | Caso Demostrativo | Tipo de Reto / Escenario | Comportamiento del Solver y Visión |
                |---|---|---|
                | 🟢 **Caso 1: Lectura Directa (4x4)** | Imagen digital nítida | Resolución inmediata en Top-1 visual (<0.2 s, sin necesidad de fallback). |
                | ⚡ **Caso 2: Rescate Neuro-Simbólico MAP (4x4)** | Ambigüedad visual en operadores | Corrección automática: la inferencia conjunta deduce el operador correcto por restricciones lógicas. |
                | 📐 **Caso 3: Perspectiva y Rotación (5x5)** | Fotografía real con ángulo e inclinación | Rectificación geométrica por homografía y corrección de perspectiva OpenCV. |
                | 🧩 **Caso 4: Alta Dificultad (6x6)** | Tablero denso y operaciones grandes | Jaulas de 3-4 celdas, multiplicaciones complejas (`240*`) y deducción combinatoria. |
                | 🏆 **Caso 5: Gran Escala (9x9)** | Tablero experto de 81 celdas | 27+ jaulas; demuestra la escalabilidad polinomial del solver CP-SAT de Google OR-Tools. |
                | ⚠️ **Caso 6: Abstención Honesta Confiable** | Degradación severa y artefactos visuales | **Política de Fidedignidad**: El sistema prefiere abstenerse (`UNRELIABLE_DETECTION`) antes que engañar al usuario con una solución inventada. |
                | ❌ **Caso 7: Tablero Infactible** | Contradicción matemática insoluble | Diagnóstico formal de infactibilidad (`INFEASIBLE`) reportado transparentemente. |
                """
            )
            gr.Examples(
                examples=valid_examples,
                example_labels=example_labels,
                inputs=[image_input, method_input, render_input],
                outputs=[image_output, metrics_output],
                fn=resolver_kenken,
                cache_examples=False,
                label="Haz clic en cualquier caso para cargarlo en la interfaz:",
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

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    )
    app = build_app()
    print(f"\n[✓] Iniciando servidor Gradio en http://{args.host}:{args.port}")
    if args.share:
        print("[✓] Generando enlace público HTTPS para acceso móvil...")
    app.launch(server_name=args.host, server_port=args.port, share=args.share, theme=theme)


if __name__ == "__main__":
    main()
