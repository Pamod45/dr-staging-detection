"""Synthetic images and results for tests - no real data, no TensorFlow."""
import json

import cv2
import numpy as np

from src import config as C
from src.decisions import Thresholds, decide
from src.inference import ScreeningResult


def fundus(h=900, w=1350, r=440, notch=True):
    """A dark-background disc like an IDRiD frame (wider than tall), optional camera notch."""
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), r, (30, 70, 150), -1)
    cv2.circle(img, (w // 2 - 150, h // 2 - 40), 18, (20, 20, 90), -1)
    if notch:
        cv2.rectangle(img, (w // 2 + r - 5, h // 2 - 8), (w // 2 + r + 12, h // 2 + 8),
                      (200, 200, 200), -1)
    return img


def fake_result(probs, referral=0.35, abstention=None):
    probs = np.asarray(probs, float)
    t = Thresholds(referral, abstention)
    rgb = np.zeros((C.MODELS[C.SCREENING_MODEL_ID].input_size,) * 2 + (3,), np.uint8)
    return ScreeningResult(
        model_id=C.SCREENING_MODEL_ID, input_size=rgb.shape[0], processed_rgb=rgb,
        overlay_rgb=rgb, probs=probs, decision=decide(probs, t), thresholds=t,
        attention={"inside_retina_pct": 91.2, "hot_area_pct": 3.4, "hot_regions": 2,
                   "hot_location": "upper left"},
        seconds=2.5)


def fake_model_folder(root, model_id=C.SCREENING_MODEL_ID, results=None):
    """Files the registry looks for; weights are dummy bytes (never loaded)."""
    d = root / model_id
    d.mkdir(parents=True, exist_ok=True)
    for f in ("best.weights.h5", "probs_test.npy", "probs_idrid.npy", "history.csv"):
        (d / f).write_bytes(b"x")
    (d / "config.json").write_text(json.dumps(
        {"IMG_SIZE": 768, "HEAD": "dense", "DROPOUT": 0.3, "UNFREEZE": 0.5, "NORM": "none"}))
    (d / "results.json").write_text(json.dumps(results or {"referral_threshold": 0.35}))
    return d
