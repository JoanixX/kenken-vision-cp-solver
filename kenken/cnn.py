"""CNN propia (PyTorch) para reconocer los caracteres de las etiquetas.

Entrada:  un glifo de 32x32 en gris, valores en [0,1] (1 = tinta). Ver ocr.normalize_glyph.
Salida:   14 puntajes ("logits"), uno por clase: 0-9, +, -, ×, ÷.
          softmax(logits) = probabilidad de cada clase.

---------------------------------------------------------------- arquitectura
    entrada                    1 x 32 x 32
    bloque 1: conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> ReLU -> maxpool2
                              32 x 16 x 16
    bloque 2: (igual)         64 x  8 x  8
    bloque 3: conv3x3 -> BN -> ReLU -> maxpool2
                             128 x  4 x  4   = 2048 valores
    clasificador: dropout -> lineal 2048->128 -> ReLU -> dropout -> lineal 128->14

  - Convolución 3x3: un filtro de 3x3 pesos que se desliza por la imagen y
    responde a un patrón local (un borde, una curva, una esquina). Cada capa
    tiene muchos filtros (32, 64, 128) y combina los patrones de la anterior:
    bordes -> trazos -> partes de dígitos (el lazo del 6, el cruce de la ×).
  - BatchNorm (BN): normaliza las activaciones de cada canal; el entrenamiento
    es más estable y tolera una tasa de aprendizaje mayor.
  - ReLU: max(0, x). La no linealidad que permite aprender formas complejas.
  - MaxPool 2x2: se queda con el máximo de cada bloque 2x2 -> reduce la
    resolución a la mitad y da tolerancia a pequeños desplazamientos.
  - Dropout: durante el entrenamiento apaga al azar una fracción de neuronas,
    lo que obliga a no depender de una sola y reduce el sobreajuste.
  - Lineal final: combina las 2048 características en 14 puntajes.

---------------------------------------------------------------- entrenamiento
  - Pérdida: entropía cruzada = -log(probabilidad asignada a la clase correcta).
    Si la red da 0.9 a la clase correcta la pérdida es 0.1; si da 0.01, es 4.6.
  - Optimizador Adam con tasa de aprendizaje que baja en coseno a lo largo de
    las épocas; cada paso usa un lote ("batch") de 128 glifos.
  - Aumentos en línea (cada época ve versiones distintas de cada glifo):
    pequeña rotación, traslación y escala, y ruido.
  - Se separa un 10 % de validación para vigilar el sobreajuste y se guarda el
    modelo de la mejor época.

---------------------------------------------------------------- datos
  Todos sintéticos, generados con la MISMA cadena de recorte/segmentación que
  usa el pipeline (ocr.label_from_gray + segment_glyphs + normalize_glyph):
    a) etiquetas sueltas al azar (render.render_label_crop) con clases balanceadas;
    b) etiquetas recortadas de tableros completos renderizados (normales y difíciles),
       pasando por toda la visión (tablero, rectificación, grilla).
  Solo se usan las etiquetas en que la segmentación dio tantos glifos como
  caracteres tiene el texto real (así cada glifo tiene su clase segura).
  La evaluación se hace sobre dataset/synthetic y dataset/synthetic_hard, que
  usan otras semillas y no se ven en el entrenamiento.

Uso:  python -m kenken.cnn            (genera datos, entrena y guarda models/ocr_cnn.pt)
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

from .ocr import (CLASS_INDEX, CLASSES, GLYPH_SIZE, GLYPH_TO_CLASS, crop_label, grid_line_mask,
                  label_from_gray, normalize_glyph, segment_glyphs)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ocr_cnn.pt"


# ============================================================ modelo
class GlyphCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()

        def conv(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]

        self.features = nn.Sequential(
            *conv(1, 32), *conv(32, 32), nn.MaxPool2d(2),     # 32x32 -> 16x16
            *conv(32, 64), *conv(64, 64), nn.MaxPool2d(2),    # 16x16 -> 8x8
            *conv(64, 128), nn.MaxPool2d(2),                  # 8x8   -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # x: (lote, 1, 32, 32)
        return self.classifier(self.features(x))


# ============================================================ datos
def glyphs_from_label(lc, text: str):
    """Glifos normalizados de una etiqueta, solo si la segmentación coincide con el texto."""
    boxes = segment_glyphs(lc.binary, lc.cell_h)
    if len(boxes) != len(text):
        return None
    return [(normalize_glyph(lc.ink, b), CLASS_INDEX[GLYPH_TO_CLASS[ch]]) for b, ch in zip(boxes, text)]


def make_label_glyphs(count: int, seed: int = 0):
    """Fuente (a): etiquetas sueltas al azar."""
    from .render import OP_GLYPHS, _has_glyph, available_fonts, random_label_text, render_label_crop
    from PIL import ImageFont

    rng = random.Random(seed)
    fonts = available_fonts()
    usable = {f: {op: [g for g in opts if not g or _has_glyph(ImageFont.truetype(f, 40), g)]
                  for op, opts in OP_GLYPHS.items()} for f in fonts}
    X, y = [], []
    for _ in range(count):
        font = rng.choice(fonts)
        glyphs = {op: rng.choice(opts or OP_GLYPHS[op][-1:]) for op, opts in usable[font].items()}
        text = random_label_text(rng, glyphs)
        out = glyphs_from_label(label_from_gray(render_label_crop(text, rng, font)), text)
        for img, cls in out or []:
            X.append(img), y.append(cls)
    return np.array(X, np.float32), np.array(y, np.int64)


def make_board_glyphs(count: int, seed: int = 0, hard_prob: float = 0.4):
    """Fuente (b): etiquetas recortadas de tableros completos por el pipeline de visión."""
    from .generator import generate
    from .pipeline import extract_structure
    from .render import render_sample

    rng = random.Random(seed)
    X, y = [], []
    for _ in range(count):
        n = rng.randint(3, 9)
        inst, sol = generate(n, seed=rng.randrange(2 ** 31))
        level = "hard" if rng.random() < hard_prob else "normal"
        img, gt = render_sample(inst, sol, rng, photo=rng.random() < 0.75, level=level)
        try:
            st = extract_structure(img)
        except ValueError:
            continue
        if st.n != n:
            continue
        lines = grid_line_mask(st.rect)
        for cage, text in zip(inst.cages, gt["image"]["labels"]):
            out = glyphs_from_label(crop_label(st.rect, lines, st.xs, st.ys, cage.anchor), text)
            for g, cls in out or []:
                X.append(g), y.append(cls)
    return np.array(X, np.float32), np.array(y, np.int64)


# ============================================================ aumentos
def augment(x: torch.Tensor) -> torch.Tensor:
    """Rotación ±10°, escala 0.85-1.15, traslación ±2.5 px, grosor de trazo y ruido."""
    b = x.shape[0]
    ang = (torch.rand(b) - 0.5) * 2 * np.deg2rad(10)
    sc = 0.85 + torch.rand(b) * 0.30
    tx, ty = (torch.rand(2, b) - 0.5) * 2 * (2.5 / GLYPH_SIZE) * 2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    # Variación no lineal de grosor de trazo (simula tinta tenue o engrosada)
    gamma = 0.75 + torch.rand(b, 1, 1, 1) * 0.60
    x = torch.clamp(x, 0.0, 1.0) ** gamma

    return (x + 0.04 * torch.randn_like(x) * torch.rand(b, 1, 1, 1)).clamp(0, 1)


# ============================================================ entrenamiento
def train(X: np.ndarray, y: np.ndarray, epochs: int = 12, batch: int = 256, lr: float = 2e-3,
          seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = len(X) // 10
    val, tr = idx[:n_val], idx[n_val:]
    Xt, yt = torch.from_numpy(X[:, None]), torch.from_numpy(y)

    # Ponderación para clases críticas de operadores (+, -, *, /)
    weights = torch.ones(len(CLASSES), dtype=torch.float32)
    for op_sym in ["+", "-", "*", "/"]:
        weights[CLASS_INDEX[op_sym]] = 1.30

    model = GlyphCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    best, best_state, history = 0.0, None, []

    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)  # orden distinto de los lotes en cada época
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for k in range(0, len(tr), batch):
            b = torch.from_numpy(tr[k:k + batch])
            xb, yb = augment(Xt[b]), yt[b]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights, label_smoothing=0.03)
            opt.zero_grad()
            loss.backward()                          # gradiente de la pérdida respecto a cada peso
            opt.step()                               # paso de optimizador
            sched.step()
            loss_sum += loss.item() * len(b)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(b)

        val_acc = evaluate_accuracy(model, Xt[val], yt[val])
        history.append({"epoch": ep + 1, "loss": loss_sum / total, "train_acc": correct / total,
                        "val_acc": val_acc, "duration_s": round(time.time() - t0, 1)})
        if verbose:
            print(f"época {ep + 1:2d}  pérdida {loss_sum / total:.4f}  "
                  f"train {correct / total:.4f}  val {val_acc:.4f}  ({time.time() - t0:.0f}s)")
        if val_acc >= best:
            best, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, history



@torch.no_grad()
def evaluate_accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = torch.cat([model(X[k:k + 1024]).argmax(1) for k in range(0, len(X), 1024)])
    return (pred == y).float().mean().item()


# ============================================================ guardar / usar
def save(model: GlyphCNN, path: str | Path = MODEL_PATH, **meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, **meta}, path)


_cache: dict = {}


def load(path: str | Path = MODEL_PATH) -> GlyphCNN:
    path = str(path)
    if path not in _cache:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["classes"] == CLASSES, "el modelo fue entrenado con otras clases"
        model = GlyphCNN()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _cache[path] = model
    return _cache[path]


@torch.no_grad()
def predict_log_probs(glyphs: np.ndarray, model: GlyphCNN | None = None) -> np.ndarray:
    """(N, 32, 32) -> (N, 14) log-probabilidades (log_softmax de los logits)."""
    if len(glyphs) == 0:
        return np.zeros((0, len(CLASSES)), np.float32)
    model = model or load()
    x = torch.from_numpy(np.asarray(glyphs, np.float32)[:, None])
    return F.log_softmax(model(x), dim=1).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera glifos sintéticos y entrena la CNN.")
    ap.add_argument("--labels", type=int, default=15000, help="etiquetas sueltas a generar")
    ap.add_argument("--boards", type=int, default=400, help="tableros completos a generar")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2)  # distinta a la de los sets de evaluación
    ap.add_argument("--out", default=str(MODEL_PATH))
    a = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    t = time.time()
    Xa, ya = make_label_glyphs(a.labels, seed=a.seed)
    print(f"(a) etiquetas sueltas: {len(Xa)} glifos  ({time.time() - t:.0f}s)")
    t = time.time()
    Xb, yb = make_board_glyphs(a.boards, seed=a.seed + 1)
    print(f"(b) recortes de tableros: {len(Xb)} glifos  ({time.time() - t:.0f}s)")
    X, y = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    print("glifos por clase:", dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist())))

    model, hist = train(X, y, epochs=a.epochs, seed=a.seed)
    save(model, a.out, history=hist, n_train=len(X), seed=a.seed)
    print(f"modelo guardado en {a.out}")

    # Guardar bitácora de entrenamiento detallada
    log_file = Path(a.out).parent.parent / "results" / "cnn_training_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"(a) etiquetas sueltas: {len(Xa)} glifos\n")
        f.write(f"(b) recortes de tableros: {len(Xb)} glifos\n")
        f.write(f"glifos por clase: {dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist()))}\n")
        for h in hist:
            f.write(f"época {h['epoch']:2d}  pérdida {h['loss']:.4f}  train {h['train_acc']:.4f}  val {h['val_acc']:.4f}  ({h.get('duration_s', 0)}s)\n")
        f.write(f"modelo guardado en {a.out}\n")

