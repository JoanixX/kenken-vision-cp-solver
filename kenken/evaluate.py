"""Métricas de evaluación.

Hito 3 (estructura): por imagen se mide
  - board_err: error máximo de esquina / lado del tablero (éxito si < 2 %)
  - n_ok:      n detectado == n real
  - cages_ok:  la partición en jaulas es exactamente la real
  - cage_f1:   fracción de jaulas reales recuperadas tal cual (acierto parcial)
  - edge_acc:  fracción de fronteras internas bien clasificadas (fina/gruesa)

Hito 4 (OCR de etiquetas), solo en jaulas cuya partición se detectó bien:
  - seg_ok:    la segmentación dio tantos glifos como caracteres reales
  - top1 / topk: la lectura correcta (objetivo y operación) es la 1.ª / está entre las k
  - inst_ok:   tablero completo correcto (estructura + todas las etiquetas en top-1)
  - inst_topk: todas las etiquetas correctas están entre sus k candidatas
               (es lo que la inferencia conjunta del modelo CP puede aprovechar)
  Además, matriz de confusión por glifo cuando la segmentación coincide.

Hito 5 (de punta a punta, imagen -> solución sin intervención):
  - solved:   el pipeline devolvió una solución
  - correct:  esa solución es la real
  - failure:  etapa a la que se atribuye el fallo:
      structure  tablero / n / jaulas mal detectados
      ocr        alguna etiqueta mal leída (el modelo CP queda INFEASIBLE o la
                 instancia es inválida)
      wrong_solution  etiqueta mal leída pero el puzzle leído SÍ tiene
                 solución: el pipeline devuelve una solución incorrecta
      model      instancia bien leída pero el solver no la resolvió

Uso:  python -m kenken.evaluate dataset/synthetic results/structure_synthetic.csv [--no-labels]
      python -m kenken.evaluate --ocr dataset/synthetic results/ocr_synthetic.csv [--no-alt]
      python -m kenken.evaluate --e2e dataset/synthetic results/e2e_synthetic.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from .ocr import CLASS_INDEX, CLASSES, GLYPH_TO_CLASS
from .pipeline import extract_structure, read_instance, solve_image
from .preprocessing import load_image


def _owner(cages, n):
    return {cell: k for k, cage in enumerate(cages) for cell in map(tuple, cage)}


def structure_metrics(gt: dict, st) -> dict:
    n = gt["n"]
    real = [sorted(map(tuple, c["cells"])) for c in gt["cages"]]
    row = {"n": n, "n_pred": st.n, "n_ok": st.n == n}
    if "image" in gt and "corners" in gt["image"]:
        c = np.array(gt["image"]["corners"])
        side = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean()
        row["board_err"] = float(np.linalg.norm(st.corners - c, axis=1).max() / side)
    pred = [sorted(c) for c in st.cages]
    row["cages_ok"] = row["n_ok"] and sorted(pred) == sorted(real)
    row["cage_f1"] = len({tuple(c) for c in pred} & {tuple(c) for c in real}) / len(real)
    if row["n_ok"]:
        ro, po = _owner(real, n), _owner(pred, n)
        edges = [((i, j), (i, j + 1)) for i in range(n) for j in range(n - 1)] + \
                [((i, j), (i + 1, j)) for i in range(n - 1) for j in range(n)]
        row["edge_acc"] = np.mean([(ro[a] == ro[b]) == (po[a] == po[b]) for a, b in edges])
    else:
        row["edge_acc"] = 0.0
    return row


def evaluate_structure(dataset_dir: str | Path, out_csv: str | Path | None = None,
                       use_labels: bool = True) -> list[dict]:
    rows = []
    for js in sorted(Path(dataset_dir).glob("*.json")):
        if js.name == "manifest.json":
            continue
        gt = json.loads(js.read_text("utf-8"))
        if "cages" not in gt:
            continue
        img_name = gt.get("image", {}).get("file")
        img_path = js.parent / img_name if img_name else next(
            p for p in js.parent.glob(js.stem + ".*") if p.suffix.lower() != ".json")
        t = time.perf_counter()
        try:
            st = extract_structure(load_image(img_path), use_labels=use_labels)
            row = structure_metrics(gt, st)
        except Exception as e:  # una imagen que falla cuenta como error, no detiene la evaluación
            row = {"n": gt["n"], "n_ok": False, "cages_ok": False, "cage_f1": 0.0,
                   "edge_acc": 0.0, "error": str(e)}
        row["time"] = time.perf_counter() - t
        row["image"] = img_path.name
        row["kind"] = gt.get("image", {}).get("kind", "real")
        rows.append(row)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = ["image", "kind", "n", "n_pred", "n_ok", "board_err", "cages_ok",
                "cage_f1", "edge_acc", "time", "error"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows


def summarize(rows: list[dict]) -> str:
    def pct(key, subset):
        return 100 * np.mean([bool(r.get(key)) for r in subset]) if subset else float("nan")

    lines = [f"{'grupo':<12}{'imgs':>6}{'tablero':>9}{'n':>8}{'jaulas':>9}{'bordes':>9}"]
    groups = [("todas", rows)] + [(k, [r for r in rows if r["kind"] == k])
                                  for k in sorted({r["kind"] for r in rows})]
    groups += [(f"n={n}", [r for r in rows if r["n"] == n]) for n in sorted({r["n"] for r in rows})]
    for name, sub in groups:
        board = 100 * np.mean([r.get("board_err", 1) < 0.02 for r in sub])
        edge = 100 * np.mean([r.get("edge_acc", 0) for r in sub])
        lines.append(f"{name:<12}{len(sub):>6}{board:>8.1f}%{pct('n_ok', sub):>7.1f}%"
                     f"{pct('cages_ok', sub):>8.1f}%{edge:>8.1f}%")
    return "\n".join(lines)


def _image_path(js: Path, gt: dict) -> Path:
    name = gt.get("image", {}).get("file")
    return js.parent / name if name else next(
        p for p in js.parent.glob(js.stem + ".*") if p.suffix.lower() != ".json")


def _gt_label_text(gt: dict, k: int) -> str | None:
    labels = gt.get("image", {}).get("labels")
    return labels[k] if labels else None


def evaluate_ocr(dataset_dir: str | Path, out_csv: str | Path | None = None, k: int = 5,
                 alternatives: bool = True, model=None):
    """Devuelve (filas por imagen, matriz de confusión de glifos 14x14)."""
    rows = []
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    for js in sorted(Path(dataset_dir).glob("*.json")):
        if js.name == "manifest.json":
            continue
        gt = json.loads(js.read_text("utf-8"))
        if "cages" not in gt:
            continue
        img_path = _image_path(js, gt)
        row = {"image": img_path.name, "kind": gt.get("image", {}).get("kind", "real"),
               "n": gt["n"], "labels": len(gt["cages"])}
        t = time.perf_counter()
        try:
            st = extract_structure(load_image(img_path))
            inst = read_instance(st, k=k, model=model, alternatives=alternatives)
        except Exception as e:
            rows.append({**row, "error": str(e), "time": time.perf_counter() - t})
            continue
        row["time"] = time.perf_counter() - t
        row["structure_ok"] = structure_metrics(gt, st)["cages_ok"]

        pred_by_cells = {tuple(sorted(c.cells)): (c, inst.candidates.get(i, []), i)
                         for i, c in enumerate(inst.cages)}
        seg_ok = top1 = topk = matched = 0
        for gk, gc in enumerate(gt["cages"]):
            key = tuple(sorted(map(tuple, gc["cells"])))
            if key not in pred_by_cells:
                continue
            matched += 1
            cage, cands, i = pred_by_cells[key]
            truth = (gc["target"], gc.get("op", "="))
            top1 += (cage.target, cage.op) == truth
            topk += any((r["target"], r["op"]) == truth for r in cands)
            text = _gt_label_text(gt, gk)
            lp = st.debug["glyph_log_probs"][i]
            if text is not None and len(lp) == len(text):
                seg_ok += 1
                for ch, row_lp in zip(text, lp):
                    confusion[CLASS_INDEX[GLYPH_TO_CLASS[ch]], int(np.argmax(row_lp))] += 1
        row.update(matched=matched, seg_ok=seg_ok, top1=top1, topk=topk,
                   inst_ok=row["structure_ok"] and top1 == len(gt["cages"]),
                   inst_topk=row["structure_ok"] and topk == len(gt["cages"]))
        rows.append(row)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = ["image", "kind", "n", "labels", "structure_ok", "matched", "seg_ok", "top1",
                "topk", "inst_ok", "inst_topk", "time", "error"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows, confusion


def summarize_ocr(rows: list[dict], confusion: np.ndarray) -> str:
    def block(name, sub):
        lab = sum(r.get("labels", 0) for r in sub)
        m = sum(r.get("matched", 0) for r in sub)
        return (f"{name:<12}{len(sub):>6}{100 * sum(r.get('seg_ok', 0) for r in sub) / max(m, 1):>9.1f}%"
                f"{100 * sum(r.get('top1', 0) for r in sub) / max(lab, 1):>8.1f}%"
                f"{100 * sum(r.get('topk', 0) for r in sub) / max(lab, 1):>8.1f}%"
                f"{100 * np.mean([bool(r.get('inst_ok')) for r in sub]):>9.1f}%"
                f"{100 * np.mean([bool(r.get('inst_topk')) for r in sub]):>10.1f}%")

    lines = [f"{'grupo':<12}{'imgs':>6}{'segment.':>10}{'top-1':>9}{'top-k':>9}{'tablero':>10}{'tab. top-k':>11}"]
    lines.append(block("todas", rows))
    for kind in sorted({r["kind"] for r in rows}):
        lines.append(block(kind, [r for r in rows if r["kind"] == kind]))
    for n in sorted({r["n"] for r in rows}):
        lines.append(block(f"n={n}", [r for r in rows if r["n"] == n]))
    acc = np.trace(confusion) / max(confusion.sum(), 1)
    lines.append(f"exactitud por glifo (segmentación correcta): {100 * acc:.2f}% "
                 f"sobre {confusion.sum()} glifos")
    return "\n".join(lines)


def _same_instance(gt: dict, inst) -> bool:
    real = {(tuple(sorted(map(tuple, c["cells"]))), c["target"], c.get("op", "=")) for c in gt["cages"]}
    pred = {(tuple(sorted(c.cells)), c.target, c.op) for c in inst.cages}
    return inst.n == gt["n"] and real == pred


def evaluate_end2end(dataset_dir: str | Path, out_csv: str | Path | None = None) -> list[dict]:
    rows = []
    for js in sorted(Path(dataset_dir).glob("*.json")):
        if js.name == "manifest.json":
            continue
        gt = json.loads(js.read_text("utf-8"))
        if "cages" not in gt:
            continue
        img_path = _image_path(js, gt)
        t = time.perf_counter()
        r = solve_image(img_path)
        row = {"image": img_path.name, "kind": gt.get("image", {}).get("kind", "real"),
               "n": gt["n"], "status": r.status, "solved": r.solved,
               "correct": r.solved and r.solution == gt.get("solution"),
               "time": time.perf_counter() - t,
               **{f"t_{k}": v for k, v in r.times.items()}}
        st_ok = r.structure is not None and structure_metrics(gt, r.structure)["cages_ok"]
        inst_ok = r.instance is not None and _same_instance(gt, r.instance)
        if row["correct"]:
            row["failure"] = ""
        elif not st_ok:
            row["failure"] = "structure"
        elif not inst_ok:
            row["failure"] = "wrong_solution" if r.solved else "ocr"
        else:
            row["failure"] = "model"
        rows.append(row)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = ["image", "kind", "n", "status", "solved", "correct", "failure", "time",
                "t_structure", "t_ocr", "t_solve", "t_render"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows


def summarize_end2end(rows: list[dict]) -> str:
    fails = ["structure", "ocr", "wrong_solution", "model"]
    lines = [f"{'grupo':<12}{'imgs':>6}{'correcta':>10}" + "".join(f"{f:>16}" for f in fails)
             + f"{'t medio':>10}"]
    groups = [("todas", rows)] + [(k, [r for r in rows if r["kind"] == k])
                                  for k in sorted({r["kind"] for r in rows})]
    groups += [(f"n={n}", [r for r in rows if r["n"] == n]) for n in sorted({r["n"] for r in rows})]
    for name, sub in groups:
        lines.append(f"{name:<12}{len(sub):>6}{100 * np.mean([r['correct'] for r in sub]):>9.1f}%"
                     + "".join(f"{100 * np.mean([r['failure'] == f for r in sub]):>15.1f}%" for f in fails)
                     + f"{np.mean([r['time'] for r in sub]):>9.2f}s")
    stages = ["t_structure", "t_ocr", "t_solve", "t_render"]
    lines.append("tiempo medio por etapa: " + ", ".join(
        f"{s[2:]}={np.mean([r[s] for r in rows if s in r]):.3f}s" for s in stages
        if any(s in r for r in rows)))
    return "\n".join(lines)


if __name__ == "__main__":
    args = []
    model_path = None
    skip_next = False
    for i, a in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if a == "--model" and i + 2 < len(sys.argv):
            model_path = sys.argv[i + 2]
            skip_next = True
            continue
        if not a.startswith("--"):
            args.append(a)

    if "--e2e" in sys.argv:
        print(summarize_end2end(evaluate_end2end(args[0], args[1] if len(args) > 1 else None)))
        sys.exit()
    if "--ocr" in sys.argv:
        rows, conf = evaluate_ocr(args[0], args[1] if len(args) > 1 else None,
                                  alternatives="--no-alt" not in sys.argv,
                                  model=model_path)
        print(summarize_ocr(rows, conf))
        if len(args) > 1:
            np.savetxt(Path(args[1]).with_suffix(".confusion.csv"), conf, fmt="%d", delimiter=",",
                       header=",".join(CLASSES), comments="")
        print(f"tiempo medio por imagen: {np.mean([r['time'] for r in rows]):.3f}s")
        sys.exit()
    rows = evaluate_structure(args[0], args[1] if len(args) > 1 else None,
                              use_labels="--no-labels" not in sys.argv)
    print(summarize(rows))
    print(f"tiempo medio por imagen: {np.mean([r['time'] for r in rows]):.3f}s")
