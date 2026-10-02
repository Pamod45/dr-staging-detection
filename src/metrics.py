"""Metrics from stored probabilities. Same definitions as the notebooks' metrics() and
referral(), in NumPy so the app needs no scikit-learn."""
import numpy as np

from src import config as C

K = C.N_CLASSES


def confusion(y: np.ndarray, yh: np.ndarray) -> np.ndarray:
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y, yh), 1)
    return cm


def qwk(y: np.ndarray, yh: np.ndarray) -> float:
    o = confusion(y, yh).astype(float)
    w = np.subtract.outer(np.arange(K), np.arange(K)) ** 2 / (K - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return float(1 - (w * o).sum() / (w * e).sum())


def recall_per_class(y: np.ndarray, yh: np.ndarray) -> np.ndarray:
    cm = confusion(y, yh)
    return cm.diagonal() / np.maximum(cm.sum(1), 1)


def macro_f1(y: np.ndarray, yh: np.ndarray) -> float:
    cm = confusion(y, yh)
    tp = cm.diagonal().astype(float)
    denom = cm.sum(0) + cm.sum(1)
    # a class with no true and no predicted cases scores 0, as sklearn's zero_division=0 does
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1.mean())


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    yh = p.argmax(1)
    out = {"Accuracy": float((yh == y).mean()), "QWK": qwk(y, yh), "Macro F1": macro_f1(y, yh)}
    out.update({f"Recall {C.LABELS[i]}": float(r) for i, r in enumerate(recall_per_class(y, yh))})
    return out


def referral(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    """Notebook 6.5: refer when P(grade 2) + P(3) + P(4) >= t."""
    truth = y >= 2
    pred = p[:, 2:].sum(1) >= t
    return {
        "sensitivity": float((truth & pred).sum() / truth.sum()),
        "specificity": float((~truth & ~pred).sum() / (~truth).sum()),
        "missed": int((truth & ~pred).sum()),
        "referred_pct": float(100 * pred.mean()),
    }


def referral_curve(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Same 0.01-0.99 grid the notebook searched."""
    return [{"threshold": float(t), **referral(y, p, t)}
            for t in np.round(np.arange(0.01, 1.0, 0.01), 2)]


def abstention(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    """Notebook 6.6: withhold the grade when the top probability is below t. The referral
    decision is separate, so referable cases among the withheld still get referred."""
    conf, yh = p.max(1), p.argmax(1)
    kept = conf >= t
    n_kept = int(kept.sum())
    return {
        "coverage": float(kept.mean()),
        "kept": n_kept,
        "withheld": int((~kept).sum()),
        "accuracy_kept": float((yh[kept] == y[kept]).mean()) if n_kept else float("nan"),
        "accuracy_all": float((yh == y).mean()),
        "qwk_kept": qwk(y[kept], yh[kept]) if n_kept > 1 else float("nan"),
        "referable_withheld_pct": (float(100 * (y[~kept] >= 2).mean()) if (~kept).any()
                                   else float("nan")),
    }


def abstention_curve(y: np.ndarray, p: np.ndarray, thresholds) -> list[dict]:
    return [{"threshold": float(t), **abstention(y, p, t)} for t in thresholds]
