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


def resolver_kenken(image: np.ndarray | None, method: str, render_mode: str):
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
    sample_files = [
        str(sample_dir / "kenken_3x3.jpg"),
        str(sample_dir / "kenken_4x4.png"),
        str(sample_dir / "kenken_5x5.jpg"),
        str(sample_dir / "kenken_6x6.jpg"),
        str(sample_dir / "kenken_9x9.png"),
    ]
    # Filtrar solo las muestras existentes
    valid_examples = [[p, "auto", "composite"] for p in sample_files if Path(p).exists()]

    with gr.Blocks(title="KenKen Vision-CP Solver") as demo:
        gr.Markdown(
            """
            # 🧩 KenKen Vision-CP Solver
            ### Sistema Híbrido Neuro-Simbólico de Visión por Computador y Programación por Restricciones
            Apunta tu **cámara web o de teléfono móvil** a un acertijo KenKen o selecciona una imagen para resolverlo en tiempo real.
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
            gr.Markdown("### 📂 Muestras Listas para Probar en 1 Clic")
            gr.Examples(
                examples=valid_examples,
                inputs=[image_input, method_input, render_input],
                outputs=[image_output, metrics_output],
                fn=resolver_kenken,
                cache_examples=False,
                label="Haz clic en cualquier muestra para probar inmediatamente:",
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
            Desarrollado con <b>OpenCV</b>, <b>PyTorch CNN</b> y <b>Google OR-Tools CP-SAT</b> | Alojado en Hugging Face Spaces
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
