# Informe Técnico Académico (`informe/`)

Este directorio contiene el manuscrito formal del proyecto redactado bajo el estándar de conferencias **IEEEtran**.

---

## Archivos Principales

- `main.tex`: Documento raíz en formato LaTeX IEEEtran.
- `informe_consolidado_IEEE.tex`: Versión consolidada monocomponente del informe.
- `refs.bib`: Base de datos de referencias bibliográficas canónicas (Google OR-Tools, CP-SAT, LCG, arquitecturas neuro-simbólicas, OpenCV).
- `IEEEtran.cls`: Clase oficial de documento de conferencias IEEE.
- `actualizar_docx.py`: Herramienta automatizada que compila el manuscrito en formato Microsoft Word (`.docx`) formal a 2 columnas con tipografía Times New Roman y fórmulas renderizadas en Office Math OMML nativo.
- `figs/`: Figuras de alta resolución empleadas directamente en el documento.

---

## Compilación del Informe en LaTeX

Para compilar el documento a PDF:

```bash
cd informe
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

O utilizando `latexmk`:

```bash
latexmk -pdf main.tex
```
