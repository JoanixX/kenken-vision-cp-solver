#!/usr/bin/env python3
"""
actualizar_docx.py
Generador de informe técnico en formato formal IEEE Conference (IEEEtran style).
Renderiza de forma nativa todas las fórmulas matemáticas de LaTeX a Office Math (OMML),
aplica tipografía formal Times New Roman, diseño a dos columnas, tablas Booktabs
sin colores informales, figuras de alta resolución incrustadas y exportación a PDF nativo.
"""

import os
import re
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

import latex2mathml.converter
from lxml import etree

FONT_NAME = "Times New Roman"

# Inicializar transformador MathML a OMML usando la plantilla oficial de Office
XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
if not os.path.exists(XSL_PATH):
    # Alternativa en x86 si correspondiera
    alt_path = r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL"
    if os.path.exists(alt_path):
        XSL_PATH = alt_path

xslt_doc = etree.parse(XSL_PATH)
transform_omml = etree.XSLT(xslt_doc)

def latex_to_omml(latex_str):
    """Convierte una cadena en formato LaTeX a un elemento XML nativo de Office Math (OMML)."""
    clean_tex = latex_str.strip()
    if clean_tex.startswith("$$") and clean_tex.endswith("$$"):
        clean_tex = clean_tex[2:-2].strip()
    elif clean_tex.startswith("$") and clean_tex.endswith("$"):
        clean_tex = clean_tex[1:-1].strip()

    # Preprocesamiento de sintaxis común
    clean_tex = clean_tex.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
    clean_tex = clean_tex.replace(r"\mathbf{1}", "1")

    try:
        mml = latex2mathml.converter.convert(clean_tex)
        mml_tree = etree.fromstring(mml)
        omml_tree = transform_omml(mml_tree)
        xml_str = etree.tostring(omml_tree.getroot(), encoding="utf-8").decode("utf-8")
        return parse_xml(xml_str)
    except Exception as e:
        # Si la conversión de sintaxis compleja falla, devolver None para fallback limpio
        return None

def set_cell_margins(cell, top=60, bottom=60, left=90, right=90):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def apply_table_booktabs_borders(table):
    """Aplica bordes formales tipo Booktabs IEEE: líneas horizontales superior, cabecera e inferior; sin líneas verticales."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D0D0D0"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def make_section_two_columns(section, spacing_pt=18):
    """Establece una sección en formato formal IEEE de dos columnas."""
    sectPr = section._sectPr
    existing_cols = sectPr.xpath('./w:cols')
    if existing_cols:
        sectPr.remove(existing_cols[0])
    spacing_dxa = int(spacing_pt * 20)
    cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="2" w:space="{spacing_dxa}"/>')
    sectPr.append(cols)

def make_section_single_column(section):
    """Establece una sección en formato de una columna."""
    sectPr = section._sectPr
    existing_cols = sectPr.xpath('./w:cols')
    if existing_cols:
        sectPr.remove(existing_cols[0])
    cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="1"/>')
    sectPr.append(cols)

def render_inline(p, text, base_size=10, italic_all=False, bold_all=False):
    """Parsea markdown simple (**bold**, *italic*, `code`, $math$, URLs) renderizando matemáticas en OMML."""
    # Tokenizar por fórmulas matemáticas, negrita, cursiva, código o URLs
    pattern = r"(\$[^\$]+\$|\*\*.*?\*\*|\*.*?\*|`.*?`|https?://[^\s\)]+)"
    tokens = re.split(pattern, text)
    for tok in tokens:
        if not tok:
            continue
        # Fórmulas matemáticas inline ($...$)
        if tok.startswith("$") and tok.endswith("$") and len(tok) > 2:
            omml_elem = latex_to_omml(tok)
            if omml_elem is not None:
                p._p.append(omml_elem)
            else:
                # Fallback tipográfico si la fórmula contiene comandos no soportados
                clean_sym = tok[1:-1].replace(r"\times", "×").replace(r"\cdot", "·").replace(r"\in", "∈").replace(r"\le", "≤").replace(r"\ge", "≥")
                run = p.add_run(clean_sym)
                run.font.name = "Cambria Math"
                run.font.size = Pt(base_size)
                run.font.italic = True
        elif tok.startswith("**") and tok.endswith("**"):
            run = p.add_run(tok[2:-2])
            run.font.name = FONT_NAME
            run.font.size = Pt(base_size)
            run.font.bold = True
            run.font.italic = italic_all
        elif tok.startswith("*") and tok.endswith("*"):
            run = p.add_run(tok[1:-1])
            run.font.name = FONT_NAME
            run.font.size = Pt(base_size)
            run.font.italic = True
            run.font.bold = bold_all
        elif tok.startswith("`") and tok.endswith("`"):
            run = p.add_run(tok[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(base_size - 0.5)
            run.font.bold = bold_all
        elif tok.startswith("http://") or tok.startswith("https://"):
            run = p.add_run(tok)
            run.font.name = FONT_NAME
            run.font.size = Pt(base_size - 0.5)
            run.font.color.rgb = RGBColor(0x00, 0x33, 0x99)
            run.font.underline = True
        else:
            run = p.add_run(tok)
            run.font.name = FONT_NAME
            run.font.size = Pt(base_size)
            run.font.italic = italic_all
            run.font.bold = bold_all

def add_figure_with_caption(doc, img_rel_path, caption_text, width_in=3.4):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, img_rel_path)
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run()
        run.add_picture(img_path, width=Inches(width_in))

        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        cp.paragraph_format.space_before = Pt(2)
        cp.paragraph_format.space_after = Pt(8)
        c_run = cp.add_run(caption_text)
        c_run.font.name = FONT_NAME
        c_run.font.size = Pt(8.5)
        c_run.font.italic = True

def render_table(doc, table_lines, caption_text=""):
    rows = []
    for line in table_lines:
        if re.match(r"^\|?\s*[-:]+[-| :]*\|?$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return

    if caption_text:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(8)
        cp.paragraph_format.space_after = Pt(3)
        c_run = cp.add_run(caption_text.upper())
        c_run.font.name = FONT_NAME
        c_run.font.size = Pt(8.5)
        c_run.font.bold = True

    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)

    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    apply_table_booktabs_borders(table)

    for r_idx, row_data in enumerate(rows):
        is_header = (r_idx == 0)
        for c_idx in range(n_cols):
            cell = table.cell(r_idx, c_idx)
            val = row_data[c_idx] if c_idx < len(row_data) else ""

            cell.text = ""
            p = cell.paragraphs[0]
            val_clean = re.sub(r"[\*`]", "", val)
            is_numeric = bool(re.match(r"^[\d\.,%\s\(\)/:\+\-\*~]+$", val_clean))
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (is_header or is_numeric) else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)

            # Renderizar celda soportando fórmulas inline si existieran
            render_inline(p, val, base_size=8.0, bold_all=is_header)

            if is_header:
                set_cell_background(cell, "F2F2F2")

            set_cell_margins(cell, top=40, bottom=40, left=70, right=70)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def build_ieee_docx(md_path, docx_path):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = docx.Document()

    # Sección 1: Cabecera formal a 1 columna
    s1 = doc.sections[0]
    s1.top_margin = Inches(0.75)
    s1.bottom_margin = Inches(1.0)
    s1.left_margin = Inches(0.625)
    s1.right_margin = Inches(0.625)
    make_section_single_column(s1)

    lines = content.split("\n")
    i = 0

    # Título (IEEE: 22 pt Bold Times New Roman)
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(12)
    t_run = title_p.add_run("Resolución Automática de KenKen Mediante Visión por Computador y Programación por Restricciones")
    t_run.font.name = FONT_NAME
    t_run.font.size = Pt(22)
    t_run.font.bold = True

    # Autores (IEEE: 11 pt Bold)
    auth_p = doc.add_paragraph()
    auth_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    auth_p.paragraph_format.space_before = Pt(0)
    auth_p.paragraph_format.space_after = Pt(2)
    a_run = auth_p.add_run("Joaquín Basas, Joaquín Alvarado, Johan Quispe")
    a_run.font.name = FONT_NAME
    a_run.font.size = Pt(11)
    a_run.font.bold = True

    # Filiación (IEEE: 9.5 pt Italic)
    aff_p = doc.add_paragraph()
    aff_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff_p.paragraph_format.space_before = Pt(0)
    aff_p.paragraph_format.space_after = Pt(2)
    aff_run = aff_p.add_run("Tópicos en Ciencias de la Computación (CC58) — Universidad Peruana de Ciencias Aplicadas (UPC), Lima, Perú")
    aff_run.font.name = FONT_NAME
    aff_run.font.size = Pt(9.5)
    aff_run.font.italic = True

    # Enlaces oficiales
    links_p = doc.add_paragraph()
    links_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    links_p.paragraph_format.space_before = Pt(2)
    links_p.paragraph_format.space_after = Pt(14)
    l1 = links_p.add_run("Repositorio GitHub: ")
    l1.font.name = FONT_NAME
    l1.font.size = Pt(9.0)
    l1.font.bold = True
    render_inline(links_p, "https://github.com/JoanixX/kenken-vision-cp-solver", base_size=9)
    l2 = links_p.add_run("   |   Demostración en Vivo: ")
    l2.font.name = FONT_NAME
    l2.font.size = Pt(9.0)
    l2.font.bold = True
    render_inline(links_p, "https://huggingface.co/spaces/joako2202/kenken-solver", base_size=9)

    # Resumen y Palabras Clave
    abs_lines = []
    kw_line = ""
    for idx, l in enumerate(lines):
        if l.startswith("### **Resumen"):
            abs_lines = lines[idx+1:idx+3]
        elif l.startswith("**Palabras clave:"):
            kw_line = lines[idx+1] if idx+1 < len(lines) else ""
            break

    abs_p = doc.add_paragraph()
    abs_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abs_p.paragraph_format.space_before = Pt(4)
    abs_p.paragraph_format.space_after = Pt(4)
    abs_p.paragraph_format.left_indent = Inches(0.4)
    abs_p.paragraph_format.right_indent = Inches(0.4)

    ab_tag = abs_p.add_run("Resumen—")
    ab_tag.font.name = FONT_NAME
    ab_tag.font.size = Pt(9.0)
    ab_tag.font.bold = True
    ab_tag.font.italic = True

    abs_text = " ".join([l.strip() for l in abs_lines if l.strip()])
    render_inline(abs_p, abs_text, base_size=9.0)

    kw_p = doc.add_paragraph()
    kw_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    kw_p.paragraph_format.space_before = Pt(2)
    kw_p.paragraph_format.space_after = Pt(14)
    kw_p.paragraph_format.left_indent = Inches(0.4)
    kw_p.paragraph_format.right_indent = Inches(0.4)

    kw_tag = kw_p.add_run("Palabras clave—")
    kw_tag.font.name = FONT_NAME
    kw_tag.font.size = Pt(9.0)
    kw_tag.font.bold = True
    kw_tag.font.italic = True
    render_inline(kw_p, kw_line.strip() if kw_line else "KenKen, visión computacional, GlyphCNN, CP-SAT, GAC, regularización L0.", base_size=9.0, italic_all=True)

    # Sección 2: Cuerpo a 2 Columnas (IEEE Standard)
    s2 = doc.add_section(docx.enum.section.WD_SECTION.CONTINUOUS)
    s2.top_margin = Inches(0.75)
    s2.bottom_margin = Inches(1.0)
    s2.left_margin = Inches(0.625)
    s2.right_margin = Inches(0.625)
    make_section_two_columns(s2, spacing_pt=18)

    body_started = False
    table_lines = []
    pending_caption = ""

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("## **I. INTRODUCCIÓN"):
            body_started = True

        if not body_started:
            i += 1
            continue

        # Acumular tabla
        if line.startswith("|") and line.endswith("|"):
            table_lines.append(line)
            i += 1
            continue
        elif table_lines:
            render_table(doc, table_lines, caption_text=pending_caption)
            table_lines = []
            pending_caption = ""

        if not line or line == "---":
            i += 1
            continue

        if line.startswith("*Tabla"):
            pending_caption = re.sub(r"[\*]", "", line).strip()
            i += 1
            continue

        # Encabezado Nivel 1 (I. SECCIÓN)
        if line.startswith("## **") or (line.startswith("## ") and re.match(r"^##\s+[IVXLCDM]+\.", line)):
            h_text = re.sub(r"[#\*]", "", line).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(h_text.upper())
            run.font.name = FONT_NAME
            run.font.size = Pt(10)
            run.font.bold = True

        # Encabezado Nivel 2 (A. Subsección)
        elif line.startswith("### **") or line.startswith("### "):
            h_text = re.sub(r"[#\*]", "", line).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(h_text)
            run.font.name = FONT_NAME
            run.font.size = Pt(10)
            run.font.italic = True
            run.font.bold = True

            # Figuras contextuales
            if "A. Arquitectura General y Rectificación" in h_text:
                add_figure_with_caption(doc, os.path.join("figs", "vision_stages.png"),
                                        "Fig. 1. Etapas del pipeline de visión computacional y rectificación proyectiva.", width_in=3.35)
            elif "C. Clasificador GlyphCNN" in h_text:
                add_figure_with_caption(doc, os.path.join("figs", "cnn_finetuning_curves.png"),
                                        "Fig. 2. Curvas de convergencia y Focal Loss durante el Fine-Tuning de GlyphCNN.", width_in=3.35)
            elif "3) Desempeño por Carácter" in h_text:
                add_figure_with_caption(doc, os.path.join("figs", "ocr_finetuned_synthetic_confusion.png"),
                                        "Fig. 3. Matriz de confusión de GlyphCNN sobre las 14 clases.", width_in=3.2)
            elif "B. Resultados Experimentales del Benchmark" in h_text:
                add_figure_with_caption(doc, os.path.join("figs", "cp_benchmark.png"),
                                        "Fig. 4. Evaluación empírica del solver CP-SAT (n=3 a 9): tiempos de pared y espacio podado.", width_in=3.35)
            elif "B. Proyección Gráfica de Soluciones" in h_text:
                add_figure_with_caption(doc, os.path.join("figs", "e2e_photo_8x8.png"),
                                        "Fig. 5. Resolución End-to-End proyectada sobre fotografía física de KenKen 8x8.", width_in=3.35)

        # Encabezado Nivel 3 (1) Sub-subsección)
        elif line.startswith("#### **") or line.startswith("#### "):
            h_text = re.sub(r"[#\*]", "", line).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(h_text)
            run.font.name = FONT_NAME
            run.font.size = Pt(9.5)
            run.font.italic = True

        # Viñetas
        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip()
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.left_indent = Inches(0.2)
            render_inline(p, bullet_text, base_size=9.5)

        # Numeración formal
        elif re.match(r"^\d+\.\s+", line):
            num_text = re.sub(r"^\d+\.\s+", "", line).strip()
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.left_indent = Inches(0.2)
            render_inline(p, num_text, base_size=9.5)

        # Ecuaciones matemáticas en bloque ($$ ... $$)
        elif line.startswith("$$") and line.endswith("$$"):
            eq_text = line[2:-2].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            omml_elem = latex_to_omml(eq_text)
            if omml_elem is not None:
                p._p.append(omml_elem)
            else:
                run = p.add_run(eq_text)
                run.font.name = "Cambria Math"
                run.font.size = Pt(9.5)
                run.font.italic = True

        # Párrafo ordinario justificado
        else:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3.5)
            p.paragraph_format.first_line_indent = Inches(0.14)
            render_inline(p, line, base_size=9.5)

        i += 1

    if table_lines:
        render_table(doc, table_lines, caption_text=pending_caption)

    doc.save(docx_path)
    print(f"[OK] Documento Word IEEE con fórmulas nativas generado en: {docx_path}")

def export_docx_to_pdf(docx_path, pdf_path):
    """Exporta a PDF nativo utilizando Microsoft Word COM automation."""
    import win32com.client
    import shutil

    docx_abs = os.path.abspath(docx_path)
    pdf_abs = os.path.abspath(pdf_path)

    temp_dir = os.environ.get("TEMP", "C:\\Temp")
    temp_docx = os.path.join(temp_dir, "ieee_kenken_build.docx")
    temp_pdf = os.path.join(temp_dir, "ieee_kenken_build.pdf")

    shutil.copyfile(docx_abs, temp_docx)

    word = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        doc = word.Documents.Open(FileName=temp_docx, ReadOnly=True)
        doc.SaveAs2(temp_pdf, FileFormat=17) # wdFormatPDF = 17
        doc.Close(False)

        shutil.copyfile(temp_pdf, pdf_abs)
        print(f"[OK] PDF con fórmulas renderizadas exportado exitosamente en: {pdf_abs} ({os.path.getsize(pdf_abs)} bytes)")
    except Exception as e:
        print(f"[ERROR] Error al exportar PDF via Word COM: {e}")
    finally:
        if word:
            try: word.Quit()
            except: pass
        if os.path.exists(temp_docx):
            try: os.remove(temp_docx)
            except: pass
        if os.path.exists(temp_pdf):
            try: os.remove(temp_pdf)
            except: pass

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(base_dir, "Quispe_Basas_Alvarado_KenKen_IEEE.docx.md")
    docx_file = os.path.join(base_dir, "Quispe_Basas_Alvarado_KenKen_IEEE.docx")
    pdf_file = os.path.join(base_dir, "Quispe_Basas_Alvarado_KenKen_IEEE.pdf")
    delivery_pdf = os.path.join(base_dir, "informe_entrega_IEEE.pdf")

    build_ieee_docx(md_file, docx_file)
    export_docx_to_pdf(docx_file, pdf_file)
    if os.path.exists(pdf_file):
        import shutil
        shutil.copyfile(pdf_file, delivery_pdf)
        shutil.copyfile(pdf_file, os.path.join(base_dir, "Quispe_Basas_Alvarado_KenKen_IEEE.docx.pdf"))
        print(f"[OK] Todas las copias de entrega sincronizadas.")
