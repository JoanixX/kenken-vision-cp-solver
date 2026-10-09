#!/usr/bin/env python3
"""Script de Fine-Tuning para GlyphCNN con Hard Example Mining y Focal Loss.

Uso:
  python scripts/finetune.py --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kenken.cnn import MODEL_PATH
from kenken.finetune import finetune, FINETUNED_MODEL_PATH, FINETUNE_LOG_PATH


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning de GlyphCNN con Focal Loss y Hard Mining")
    parser.add_argument("--epochs", type=int, default=8, help="Número de épocas de ajuste fino")
    parser.add_argument("--base", default=str(MODEL_PATH), help="Ruta al modelo base a partir del cual ajustar")
    parser.add_argument("--out", default=str(FINETUNED_MODEL_PATH), help="Ruta donde guardar el modelo fine-tuned")
    parser.add_argument("--lr", type=float, default=3e-4, help="Tasa de aprendizaje")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Descongelar toda la red desde el inicio")
    parser.add_argument("--device", default=None, help="Dispositivo (cpu / cuda)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Iniciando fine-tuning de {args.base} -> {out_path} ({args.epochs} épocas)...")
    tuned_model, history = finetune(
        base_model_path=args.base,
        out_path=out_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        freeze_conv1=not args.unfreeze_backbone,
        device=args.device,
    )

    # Guardar bitácora en results/logs/ y en results/
    log_path_logs = ROOT_DIR / "results" / "logs" / "cnn_finetuning_log.txt"
    log_path_logs.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path_logs, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_acc\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']:.5f},{h['val_acc']:.4f}\n")

    # Copia de compatibilidad en results/
    if FINETUNE_LOG_PATH.exists():
        with open(FINETUNE_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_acc\n")
            for h in history:
                f.write(f"{h['epoch']},{h['train_loss']:.5f},{h['val_acc']:.4f}\n")

    print(f"[OK] Modelo guardado en: {out_path}")
    print(f"[OK] Bitácora guardada en: {log_path_logs}")


if __name__ == "__main__":
    main()
