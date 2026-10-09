"""Build an entirely static site from the repository and real pipeline results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT / "web" / "public"
CATEGORIES = {
    "1": ("Lectura directa", "Etiquetas nítidas y resolución directa."),
    "2": ("Rescate MAP", "Lecturas ambiguas que requieren consistencia matemática."),
    "3": ("Perspectiva", "Fotos inclinadas y corrección geométrica."),
    "4": ("Gran escala", "Tableros grandes y operaciones de mayor complejidad."),
    "5": ("Estrés", "Varias etiquetas degradadas en una misma imagen."),
    "6": ("Múltiples ajustes", "Revisa las etiquetas corregidas antes de confiar en el resultado."),
    "7": ("Contradicciones", "Casos diseñados para explorar la detección de infactibilidad."),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), "utf-8")


def build(precompute=False):
    assets = PUBLIC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sources = {}
    for folder in ["kenken", "scripts", "tests", "docs", "web", "prototipo_interactivo"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".js", ".css", ".html"} and "__pycache__" not in path.parts:
                sources[path.relative_to(ROOT).as_posix()] = path.read_text("utf-8", errors="replace")
    for name in ["README.md", "pyproject.toml", "requirements.txt"]:
        sources[name] = (ROOT / name).read_text("utf-8")
    sources = dict(sorted(sources.items()))
    write_json(PUBLIC / "data" / "sources.json", sources)
    write_json(PUBLIC / "data" / "examples.json", {p.name: json.loads(p.read_text("utf-8")) for p in sorted((ROOT / "examples").glob("*.json"))})
    cases = []
    for path in sorted((ROOT / "prototipo_interactivo" / "sample_images").glob("tipo*")):
        category, description = CATEGORIES[path.name[4]]
        shutil.copy2(path, assets / path.name)
        result_path = PUBLIC / "data" / f"{path.stem}.json"
        case = {"id": path.stem, "category": category, "description": description,
                "title": path.stem.split("_", 1)[1].replace("_", " ").capitalize().replace("x", " × "),
                "image": f"assets/{path.name}", "result": f"data/{path.stem}.json" if result_path.exists() else None}
        if precompute:
            from kenken.pipeline import solve_image
            print(f"Precalculando {path.name}", flush=True)
            start = time.perf_counter()
            result = solve_image(path, method="auto", time_limit=10)
            solver = result.solve_result
            payload = {"status": result.status, "grid": result.grid, "instance": result.instance.to_dict(),
                       "fallback_used": result.fallback_used, "num_changed": result.num_changed,
                       "fidelity": result.fidelity, "confidence_level": result.confidence_level,
                       "divergent": result.divergent, "solver_ms": round(solver.wall_time * 1000, 2),
                       "branches": solver.branches, "conflicts": solver.conflicts,
                       "total_ms": round((time.perf_counter() - start) * 1000, 2),
                       "engine": "Python · OpenCV + GlyphCNN + CP-SAT", "images": {}}
            if result.solved:
                for mode in ["clean", "original", "rectified", "composite"]:
                    name = f"{path.stem}-{mode}.png"
                    result.render_solution(out_path=assets / name, mode=mode)
                    payload["images"][mode] = f"assets/{name}"
            write_json(result_path, payload)
            case["result"] = f"data/{path.stem}.json"
        cases.append(case)
    write_json(PUBLIC / "data" / "manifest.json", {"cases": cases, "sources": list(sources)})
    # Rev the service worker on content changes. Assets are cached on first use.
    import hashlib
    revision = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(PUBLIC.rglob("*")) if p.is_file() and p.name != "sw.js")).hexdigest()[:12]
    (PUBLIC / "sw.js").write_text("const CACHE='kenken-" + revision + "';\n" + (ROOT / "web" / "sw-template.js").read_text("utf-8"), "utf-8")
    (PUBLIC / ".nojekyll").touch()
    print(f"Web lista: {len(cases)} casos, {len(sources)} fuentes. {PUBLIC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", action="store_true", help="Recalcular las fotos para el sitio estático")
    build(parser.parse_args().precompute)
