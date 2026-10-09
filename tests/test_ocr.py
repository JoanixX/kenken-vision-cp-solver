"""Pruebas del hito 4: segmentación de etiquetas, decodificación y CNN."""

import random

import numpy as np
import pytest
import torch

from kenken.cnn import MODEL_PATH, GlyphCNN
from kenken.ocr import (CLASS_INDEX, CLASSES, GLYPH_SIZE, decode_readings, label_from_gray,
                        normalize_glyph, segment_glyphs)
from kenken.render import available_fonts, render_label_crop


def one_hot_log_probs(text: str, confidence: float = 0.9) -> np.ndarray:
    """Log-probs falsas: la clase correcta con `confidence`, el resto repartido."""
    lp = np.full((len(text), len(CLASSES)), np.log((1 - confidence) / (len(CLASSES) - 1)))
    for p, ch in enumerate(text):
        lp[p, CLASS_INDEX[ch]] = np.log(confidence)
    return lp


@pytest.mark.parametrize("text", ["7+", "12*", "3-", "2/", "240*"])
def test_segmentation_counts_characters(text):
    font = next(f for f in available_fonts() if "arial" in f.lower() or "DejaVuSans" in f)
    glyphs = {"*": "×", "/": "÷", "-": "−"}
    shown = "".join(glyphs.get(c, c) for c in text)
    lc = label_from_gray(render_label_crop(shown, random.Random(0), font))
    boxes = segment_glyphs(lc.binary, lc.cell_h)
    assert len(boxes) == len(text)
    assert all(normalize_glyph(lc.ink, b).shape == (GLYPH_SIZE, GLYPH_SIZE) for b in boxes)


def test_decode_best_reading():
    r = decode_readings(one_hot_log_probs("12*"), cage_size=3, n=6)
    assert (r[0]["target"], r[0]["op"]) == (12, "*")
    assert r == sorted(r, key=lambda x: -x["logp"])


def test_decode_respects_cage_size():
    # Jaula de 3 celdas: '-' ni '/' están permitidos aunque la CNN los prefiera.
    r_sub = decode_readings(one_hot_log_probs("4-"), cage_size=3, n=6)
    assert all(x["op"] in "+*" for x in r_sub)
    r_div = decode_readings(one_hot_log_probs("2/"), cage_size=3, n=6)
    assert all(x["op"] in "+*" for x in r_div)
    # Jaula de 1 celda: sin operación, todos los glifos son dígitos.
    r = decode_readings(one_hot_log_probs("5"), cage_size=1, n=6)
    assert (r[0]["target"], r[0]["op"]) == (5, "=")


def test_decode_discards_impossible_targets():
    # '÷' con objetivo 8 en un 6x6 es imposible -> no debe aparecer.
    r = decode_readings(one_hot_log_probs("8/"), cage_size=2, n=6)
    assert all(not (x["op"] == "/" and x["target"] > 6) for x in r)


def test_cnn_output_shape():
    out = GlyphCNN()(torch.zeros(3, 1, GLYPH_SIZE, GLYPH_SIZE))
    assert tuple(out.shape) == (3, len(CLASSES))


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="modelo no entrenado")
def test_trained_model_reads_clean_labels():
    from kenken.cnn import predict_log_probs

    rng = random.Random(5)
    font = next(f for f in available_fonts() if "arial" in f.lower() or "DejaVuSans" in f)
    ok = 0
    for text, shown in [("15+", "15+"), ("6*", "6×"), ("2/", "2÷"), ("3-", "3−"), ("48*", "48x")]:
        lc = label_from_gray(render_label_crop(shown, rng, font))
        boxes = segment_glyphs(lc.binary, lc.cell_h)
        glyphs = np.stack([normalize_glyph(lc.ink, b) for b in boxes])
        pred = "".join(CLASSES[k] for k in predict_log_probs(glyphs).argmax(1))
        ok += pred == text
    assert ok >= 4
