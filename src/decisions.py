import json
from dataclasses import dataclass

import numpy as np

from src import config as C
from src import registry


@dataclass(frozen=True)
class Thresholds:
    referral: float
    abstention: float | None


@dataclass(frozen=True)
class Decision:
    grade: int
    confidence: float
    referable_prob: float
    refer: bool
    abstained: bool


def load_thresholds(model_id: str = C.SCREENING_MODEL_ID) -> Thresholds:
    path = registry.model_dir(model_id) / "results.json"
    res = json.loads(path.read_text())
    if "referral_threshold" not in res:
        raise KeyError(f"{path} has no referral_threshold")
    abst = res.get("abstention_threshold")
    return Thresholds(float(res["referral_threshold"]), float(abst) if abst is not None else None)


def decide(p: np.ndarray, t: Thresholds) -> Decision:
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (C.N_CLASSES,):
        raise ValueError(f"expected {C.N_CLASSES} probabilities, got shape {p.shape}")
    grade, conf = int(p.argmax()), float(p.max())
    referable = float(p[list(C.REFERABLE_GRADES)].sum())
    return Decision(
        grade=grade,
        confidence=conf,
        referable_prob=referable,
        refer=referable >= t.referral,
        abstained=t.abstention is not None and conf < t.abstention,
    )
