"""Módulo de Fine-Tuning para GlyphCNN con Hard Example Mining y Focal Loss.

Preserva el modelo base original (models/ocr_cnn.pt) intacto y genera una versión
mejorada (models/ocr_cnn_finetuned.pt) optimizada para pares de caracteres
confusos (+ vs *, / vs +, 8 vs 5) y perturbaciones físicas severas.

Uso:
    python -m kenken.finetune --epochs 8 --base models/ocr_cnn.pt --out models/ocr_cnn_finetuned.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import (CLASS_INDEX, CLASSES, GLYPH_SIZE, MODEL_PATH, GlyphCNN,
                  augment, evaluate_accuracy, glyphs_from_label, load, save)
from .ocr import crop_label, grid_line_mask, label_from_gray


FINETUNED_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ocr_cnn_finetuned.pt"
FINETUNE_LOG_PATH = Path(__file__).resolve().parent.parent / "results" / "cnn_finetuning_log.txt"


# ============================================================ 1. pérdida focal
class MultiClassFocalLoss(nn.Module):
    """Focal Loss para problemas multiclase con ponderación de clases conflictivas.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focaliza el gradiente en las muestras difíciles o ambiguas (p_t bajo),
    impidiendo que las muestras fáciles dominen la actualización.
    """

    def __init__(self, gamma: float = 1.5, weight: torch.Tensor | None = None, label_smoothing: float = 0.02):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================ 2. minería de datos
def make_hard_crops(count: int, seed: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Genera recortes de etiquetas con alta degradación y sobremuestreo de clases conflictivas."""
    from PIL import ImageFont
    from .render import OP_GLYPHS, _has_glyph, available_fonts, random_label_text, render_label_crop

    rng = random.Random(seed)
    fonts = available_fonts()
    usable = {
        f: {op: [g for g in opts if not g or _has_glyph(ImageFont.truetype(f, 40), g)]
            for op, opts in OP_GLYPHS.items()}
        for f in fonts
    }

    # Pares que sufren mayor confusión en los benchmarks: operadores y dígitos ambiguos
    priority_ops = ["/", "+", "-", "*"]
    X, y = [], []

    for i in range(count):
        font = rng.choice(fonts)
        # 60% de las veces forzar operadores conflictivos
        if rng.random() < 0.60:
            target_op = rng.choice(priority_ops)
            glyphs = {op: rng.choice(opts or OP_GLYPHS[op][-1:]) for op, opts in usable[font].items()}
            # Construir texto con target_op y dígitos que suelen confundirse (8, 5, 2, 1, 4)
            d1 = rng.choice(["1", "2", "4", "5", "8", "3", "6", "7", "9"])
            d2 = rng.choice(["", "2", "4", "5", "8"])
            text = f"{d1}{d2}{glyphs[target_op]}"
        else:
            glyphs = {op: rng.choice(opts or OP_GLYPHS[op][-1:]) for op, opts in usable[font].items()}
            text = random_label_text(rng, glyphs)

        crop = render_label_crop(text, rng, font)
        lc = label_from_gray(crop)
        out = glyphs_from_label(lc, text)
        for img, cls in out or []:
            X.append(img)
            y.append(cls)

    return np.array(X, np.float32), np.array(y, np.int64)


def make_hard_board_glyphs(count: int, seed: int = 202) -> tuple[np.ndarray, np.ndarray]:
    """Extrae glifos de tableros renderizados con perturbaciones de nivel difícil."""
    from .generator import generate
    from .pipeline import extract_structure
    from .render import render_sample

    rng = random.Random(seed)
    X, y = [], []
    for idx in range(count):
        if (idx + 1) % 10 == 0 or idx == count - 1:
            print(f"    ... procesando tablero difícil {idx + 1}/{count}", flush=True)
        n = rng.randint(3, 9)
        inst, sol = generate(n, seed=rng.randrange(2 ** 31))
        # 85% nivel difícil con degradación física de iluminación y perspectiva
        img, gt = render_sample(inst, sol, rng, photo=True, level="hard")
        try:
            st = extract_structure(img)
        except Exception:
            continue
        if st.n != n:
            continue
        lines = grid_line_mask(st.rect)
        for cage, text in zip(inst.cages, gt.get("image", {}).get("labels", [])):
            try:
                lc = crop_label(st.rect, lines, st.xs, st.ys, cage.anchor)
                out = glyphs_from_label(lc, text)
                for g, cls in out or []:
                    X.append(g)
                    y.append(cls)
            except Exception:
                continue

    return np.array(X, np.float32), np.array(y, np.int64)


# ============================================================ 3. ciclo de fine-tuning
def finetune(
    base_model: GlyphCNN,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 8,
    batch: int = 256,
    lr: float = 2e-4,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[GlyphCNN, list[dict]]:
    """Ajuste fino de la CNN partiendo de los pesos base."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(100, len(X) // 10)
    val, tr = idx[:n_val], idx[n_val:]
    Xt, yt = torch.from_numpy(X[:, None]), torch.from_numpy(y)

    # Ponderación específica para operadores conflictivos
    weights = torch.ones(len(CLASSES), dtype=torch.float32)
    weights[CLASS_INDEX["/"]] = 1.40
    weights[CLASS_INDEX["+"]] = 1.35
    weights[CLASS_INDEX["-"]] = 1.25
    weights[CLASS_INDEX["*"]] = 1.25
    loss_fn = MultiClassFocalLoss(gamma=1.5, weight=weights, label_smoothing=0.02)

    model = base_model
    # Congelar el Bloque 1 inicial en las primeras 2 épocas para proteger los filtros de bajo nivel
    for p in model.features[:6].parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    best_acc = evaluate_accuracy(model, Xt[val], yt[val])
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = []

    if verbose:
        print(f"Exactitud inicial con pesos base antes del fine-tuning: {best_acc:.4f}")

    for ep in range(epochs):
        # Descongelar todas las capas a partir de la época 3 para adaptación conjunta
        if ep == 2:
            for p in model.features[:6].parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.75, weight_decay=1e-4)
            remaining_steps = (epochs - ep) * int(np.ceil(len(tr) / batch))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining_steps, eta_min=1e-5)

        model.train()
        rng.shuffle(tr)
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0

        for k in range(0, len(tr), batch):
            b = torch.from_numpy(tr[k:k + batch])
            xb, yb = augment(Xt[b]), yt[b]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(b)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(b)

        val_acc = evaluate_accuracy(model, Xt[val], yt[val])
        ep_duration = round(time.time() - t0, 1)
        history.append({
            "epoch": ep + 1,
            "loss": loss_sum / total,
            "train_acc": correct / total,
            "val_acc": val_acc,
            "duration_s": ep_duration,
        })
        if verbose:
            print(f"época {ep + 1:2d}  pérdida {loss_sum / total:.4f}  "
                  f"train {correct / total:.4f}  val {val_acc:.4f}  ({ep_duration}s)")

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, history


# ============================================================ 4. main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta Fine-Tuning sobre GlyphCNN.")
    parser.add_argument("--base", default=str(MODEL_PATH), help="Ruta al modelo base preentrenado")
    parser.add_argument("--out", default=str(FINETUNED_MODEL_PATH), help="Ruta donde guardar el modelo fine-tuned")
    parser.add_argument("--hard-labels", type=int, default=12000, help="Etiquetas difíciles a generar")
    parser.add_argument("--hard-boards", type=int, default=150, help="Tableros difíciles a generar")
    parser.add_argument("--epochs", type=int, default=8, help="Épocas de fine-tuning")
    parser.add_argument("--lr", type=float, default=2e-4, help="Tasa de aprendizaje")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    print(f"Cargando modelo base desde: {args.base}", flush=True)
    base_model = load(args.base)

    t0 = time.time()
    print("Generando dataset especializado con Hard Example Mining...", flush=True)
    X_labels, y_labels = make_hard_crops(args.hard_labels, seed=args.seed)
    print(f"  (a) Etiquetas difíciles minadas: {len(X_labels)} glifos ({time.time() - t0:.1f}s)", flush=True)

    t1 = time.time()
    X_boards, y_boards = make_hard_board_glyphs(args.hard_boards, seed=args.seed + 1)
    print(f"  (b) Glifos de tableros difíciles: {len(X_boards)} glifos ({time.time() - t1:.1f}s)", flush=True)

    X = np.concatenate([X_labels, X_boards]) if len(X_boards) else X_labels
    y = np.concatenate([y_labels, y_boards]) if len(y_boards) else y_labels
    print(f"Total glifos para fine-tuning: {len(X)}", flush=True)
    print("Distribución por clase:", dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist())), flush=True)

    print("\nIniciando ciclo de Fine-Tuning con Focal Loss...", flush=True)
    tuned_model, history = finetune(
        base_model, X, y, epochs=args.epochs, lr=args.lr, seed=args.seed, verbose=True
    )

    save(tuned_model, args.out, history=history, n_train=len(X), seed=args.seed, base_model=str(args.base))
    print(f"\n[OK] Modelo Fine-Tuned guardado en: {args.out}", flush=True)

    # Guardar bitácora
    FINETUNE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINETUNE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"Modelo base: {args.base}\n")
        f.write(f"Glifos totales: {len(X)}\n")
        f.write(f"Distribución: {dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist()))}\n")
        for h in history:
            f.write(f"época {h['epoch']:2d}  pérdida {h['loss']:.4f}  train {h['train_acc']:.4f}  val {h['val_acc']:.4f}  ({h['duration_s']}s)\n")
        f.write(f"Guardado en: {args.out}\n")
    print(f"Bitácora guardada en: {FINETUNE_LOG_PATH}")
