"""Readers for everything the Results and Comparison pages show: stored probabilities, labels,
training histories and the saved run summaries. File reads only - no model is loaded."""
import json

import numpy as np
import pandas as pd

from src import config as C
from src import registry

V1_ARTEFACTS = C.ROOT / "notebooks" / "dr_grading_kaggle_v1" / "artefacts"
STRATEGY_HISTORY = V1_ARTEFACTS / "strategy" / "strategy_history.json"

# results.json keys use the notebook's label spelling
_NB_LABELS = ("No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR")


def labels(model_id: str) -> np.ndarray:
    spec = C.MODELS[model_id]
    return np.load(C.LABEL_FILES[(spec.notebook, spec.eval_set)])


def probs(model_id: str) -> np.ndarray:
    spec = C.MODELS[model_id]
    return np.load(registry.model_dir(model_id) / spec.probs_file)


def history(model_id: str) -> pd.DataFrame:
    """One row per epoch, 'epoch' counted from 1, for either notebook's format."""
    spec = C.MODELS[model_id]
    path = registry.model_dir(model_id) / spec.history_file
    if path.suffix == ".csv":
        return pd.read_csv(path)
    df = pd.DataFrame(json.loads(path.read_text()))
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    return df


def run_results(model_id: str) -> dict:
    return json.loads((registry.model_dir(model_id) / "results.json").read_text())


def stored_summary(model_id: str, split: str) -> dict | None:
    """Metrics the notebook saved for a split it did not save probabilities for (IDRiD).
    Returns None when that split is absent from results.json."""
    res = run_results(model_id)
    if f"{split}_accuracy" not in res:
        return None
    out = {"Accuracy": res[f"{split}_accuracy"], "QWK": res[f"{split}_QWK"],
           "Macro F1": res[f"{split}_macro F1"]}
    out.update({f"Recall {C.LABELS[i]}": res[f"{split}_recall {nb}"]
                for i, nb in enumerate(_NB_LABELS)})
    return out


def strategy_histories() -> dict[str, pd.DataFrame]:
    """v1 section 5.1 runs (fixed lr, plateau, cosine). Their weights are not in models/,
    only this history file is needed."""
    raw = json.loads(STRATEGY_HISTORY.read_text())
    out = {}
    for name, h in raw.items():
        df = pd.DataFrame(h)
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
        out[name] = df
    return out


# Fallback when labels_idrid.npy is absent: the support column of the v2 notebook's IDRiD
# classification report (section 6.2), after the 27 label corrections.
IDRID_COUNTS = {0: 118, 1: 23, 2: 164, 3: 90, 4: 60}


def split_arrays(model_id: str, split: str):
    """(labels, probs) the notebook saved for one split ("val", "test" or "idrid"), or None
    when either file is missing."""
    folder = registry.model_dir(model_id)
    probs_file, labels_file = folder / f"probs_{split}.npy", folder / f"labels_{split}.npy"
    if split == "test" and not labels_file.exists():
        spec = C.MODELS[model_id]
        labels_file = C.LABEL_FILES[(spec.notebook, spec.eval_set)]
    if not (probs_file.exists() and labels_file.exists()):
        return None
    return np.load(labels_file), np.load(probs_file)


def idrid_counts(model_id: str = C.SCREENING_MODEL_ID) -> dict:
    arrays = split_arrays(model_id, "idrid")
    if arrays is None:
        return IDRID_COUNTS
    counts = np.bincount(arrays[0], minlength=C.N_CLASSES)
    return {g: int(n) for g, n in enumerate(counts)}


def ddr_grade_counts() -> pd.DataFrame:
    """Images per grade and split in the final notebook's DDR split."""
    split = pd.read_csv(C.SPLIT_FILES["v2"])
    return split.groupby(["diagnosis", "split"]).size().rename("images").reset_index()
