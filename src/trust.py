"""How far to trust one result, measured on the screening model's own test predictions
(probs_test.npy against test_labels_v2.npy, 1,788 DDR images). Arithmetic on stored arrays
only; nothing here runs a model or estimates anything that was not measured."""
from functools import lru_cache

import numpy as np

from src import config as C
from src import data, registry
from src.explanation import pct

BANDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@lru_cache(maxsize=4)
def _arrays(model_id: str, mtime: float):
    return data.labels(model_id), data.probs(model_id)


def arrays(model_id: str):
    """(labels, probs), or None when the stored test files are missing."""
    if registry.check_model(model_id)["status"] != "ok":
        return None
    path = registry.model_dir(model_id) / C.MODELS[model_id].probs_file
    try:
        return _arrays(model_id, path.stat().st_mtime)
    except (OSError, ValueError, KeyError):
        return None


def band(conf: float) -> tuple[float, float]:
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        if conf < hi or hi == BANDS[-1]:
            return lo, hi
    return BANDS[-2], BANDS[-1]


def reliability(y: np.ndarray, p: np.ndarray, grade: int, conf: float, refer: bool,
                threshold: float) -> dict:
    pred = p.argmax(1)
    conf_all = p.max(1)

    same_call = pred == grade
    truths = np.bincount(y[same_call], minlength=C.N_CLASSES)

    lo, hi = band(conf)
    in_band = (conf_all >= lo) & ((conf_all < hi) | (hi == BANDS[-1]))
    band_n = int(in_band.sum())

    truth_ref = y >= 2
    called_ref = p[:, 2:].sum(1) >= threshold
    same_ref = called_ref == refer

    return {
        "n_test": int(len(y)),
        "grade_calls": int(same_call.sum()),
        "grade_right": int(truths[grade]),
        "grade_truths": {int(g): int(n) for g, n in enumerate(truths) if n},
        "band": (lo, hi),
        "band_n": band_n,
        "band_right": int((pred[in_band] == y[in_band]).sum()),
        "band_within_one": int((np.abs(pred[in_band] - y[in_band]) <= 1).sum()),
        "referral_calls": int(same_ref.sum()),
        "referral_right": int((truth_ref[same_ref] == refer).sum()),
    }


def reliability_text(result) -> str | None:
    """The RELIABILITY block for the chat, or None if the test files are unavailable."""
    got = arrays(result.model_id)
    if got is None:
        return None
    y, p = got
    d = result.decision
    r = reliability(y, p, d.grade, d.confidence, d.refer, result.thresholds.referral)
    lines = [f"Measured on the {r['n_test']:,} DDR test images, which played no part in training "
             f"or in choosing thresholds."]
    if not d.abstained:
        others = ", ".join(f"{C.LABELS[g]} {n}" for g, n in r["grade_truths"].items()
                           if g != d.grade)
        lines.append(
            f"When the model's top grade was {C.LABELS[d.grade]}: {r['grade_calls']} test images; "
            f"{r['grade_right']} were truly {C.LABELS[d.grade]} "
            f"({pct(r['grade_right'] / max(r['grade_calls'], 1))}). The others were truly: "
            f"{others or 'none'}.")
    lo, hi = r["band"]
    lines.append(
        f"Test images graded with confidence between {pct(lo)} and {pct(hi)} (this image: "
        f"{pct(d.confidence)}): {r['band_n']}; top grade exactly right for {r['band_right']} "
        f"({pct(r['band_right'] / max(r['band_n'], 1))}), within one grade for "
        f"{r['band_within_one']} ({pct(r['band_within_one'] / max(r['band_n'], 1))}).")
    call = "referred" if d.refer else "not referred"
    truth = "truly referable (Moderate or worse)" if d.refer else "truly not referable"
    lines.append(
        f"Test images {call} at this threshold: {r['referral_calls']}; {r['referral_right']} were "
        f"{truth} ({pct(r['referral_right'] / max(r['referral_calls'], 1))}).")
    return "\n".join(lines)
