"""Figuras para el informe a partir de los resultados guardados en results/.

Uso:  python -m kenken.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .cnn import MODEL_PATH  # noqa: E402
from .ocr import CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs"
SHOWN = list("0123456789") + ["+", "−", "×", "÷"]


def training_curves(out=FIGS / "cnn_training.png"):
    hist = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)["history"]
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.plot(ep, [h["loss"] for h in hist], "o-")
    a.set_xlabel("época"), a.set_ylabel("entropía cruzada (train)"), a.grid(alpha=0.3)
    b.plot(ep, [100 * h["train_acc"] for h in hist], "o-", label="entrenamiento (con aumentos)")
    b.plot(ep, [100 * h["val_acc"] for h in hist], "s-", label="validación")
    b.set_xlabel("época"), b.set_ylabel("exactitud (%)"), b.legend(), b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def confusion_matrix(csv_path, out, title):
    conf = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), SHOWN)
    ax.set_yticks(range(len(CLASSES)), SHOWN)
    ax.set_xlabel("predicho"), ax.set_ylabel("real")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if conf[i, j] and (i != j or norm[i, j] < 0.995):
                ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    acc = np.trace(conf) / conf.sum()
    ax.set_title(f"{title}: {100 * acc:.2f}% de {int(conf.sum())} glifos")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    training_curves()
    for name, title in [("ocr_synthetic", "Sintético normal"), ("ocr_synthetic_hard", "Sintético difícil")]:
        csv = ROOT / "results" / f"{name}.confusion.csv"
        if csv.exists():
            confusion_matrix(csv, FIGS / f"{name}_confusion.png", title)
    print("figuras en", FIGS)
