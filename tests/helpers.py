"""Synthetic images and results for tests."""
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


V2_TRAINING_CONFIG = {
    "ROTATION": 360, "ZOOM": [0.9, 1.1], "BRIGHTNESS": 0.2, "CONTRAST": 0.1, "CHANNEL": 0.05,
    "PHASE1_EPOCHS": 5, "PHASE1_LR": 1e-3, "PHASE2_LR": 1e-4, "RLR_FACTOR": 0.3,
    "RLR_PATIENCE": 2, "ES_PATIENCE": 5,
}


def fake_full_store(root, monkeypatch, seed=1):
    """Every model in config.MODELS with synthetic probabilities, histories and run files,
    plus label arrays. Weights are dummy bytes: nothing here loads a model."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(C, "MODELS_DIR", root)
    label_files = {}
    for key, n in {("v1", "val"): 1792, ("v2", "test"): 1788}.items():
        f = root / f"labels_{key[0]}_{key[1]}.npy"
        np.save(f, rng.integers(0, 5, n))
        label_files[key] = f
    monkeypatch.setattr(C, "LABEL_FILES", label_files)

    for m, spec in C.MODELS.items():
        d = root / m
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.weights.h5").write_bytes(b"x")
        epochs = int(rng.integers(8, 14))
        qwk = [round(0.7 + 0.01 * e + rng.uniform(0, 0.01), 4) for e in range(epochs)]
        if spec.notebook == "v1":
            y = np.load(label_files[("v1", "val")])
            p = np.eye(5)[y] * rng.uniform(0.3, 0.7) + rng.dirichlet(np.ones(5), len(y))
            np.save(d / "val_probs.npy", p / p.sum(1, keepdims=True))
            (d / "history.json").write_text(json.dumps({"val_qwk": qwk, "loss": qwk}))
            continue
        (d / "history.csv").write_text("epoch,val_qwk,val_accuracy\n" + "\n".join(
            f"{e + 1},{q},{q}" for e, q in enumerate(qwk)))
        (d / "config.json").write_text(json.dumps(
            {"IMG_SIZE": spec.input_size, "CACHE_SIZE": 768, "HEAD": "dense", "DROPOUT": 0.3,
             "UNFREEZE": 0.5, "NORM": "none", "LOSS": "focal", "FOCAL_GAMMA": 2.0,
             "MERGE_EXTRA": "merged" in m, **V2_TRAINING_CONFIG}))
        (d / "results.json").write_text(json.dumps(
            {"referral_threshold": 0.35, "train_images": 2000 if "2000" in m else 8949,
             "epochs_run": epochs, "best_epoch": epochs - 2, "phase2_minutes": 40.0,
             "idrid_accuracy": 0.66, "idrid_QWK": 0.76, "idrid_macro F1": 0.6,
             **{f"idrid_recall {n}": 0.5 for n in
                ("No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR")}}))
        if spec.probs_file:
            y = np.load(label_files[("v2", "test")])
            p = np.eye(5)[y] * rng.uniform(0.3, 0.7) + rng.dirichlet(np.ones(5), len(y))
            np.save(d / "probs_test.npy", p / p.sum(1, keepdims=True))
            (d / "probs_idrid.npy").write_bytes(b"x")
    return root
