"""Métricas de evaluación.

Hito 3 (estructura): por imagen se mide
  - board_err: error máximo de esquina / lado del tablero (éxito si < 2 %)
  - n_ok:      n detectado == n real
  - cages_ok:  la partición en jaulas es exactamente la real
  - cage_f1:   fracción de jaulas reales recuperadas tal cual (acierto parcial)
  - edge_acc:  fracción de fronteras internas bien clasificadas (fina/gruesa)

Uso:  python -m kenken.evaluate dataset/synthetic results/structure_synthetic.csv [--no-labels]
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from .pipeline import extract_structure
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
        gt = json.loads(js.read_text("utf-8"))
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


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = evaluate_structure(args[0], args[1] if len(args) > 1 else None,
                              use_labels="--no-labels" not in sys.argv)
    print(summarize(rows))
    print(f"tiempo medio por imagen: {np.mean([r['time'] for r in rows]):.3f}s")
