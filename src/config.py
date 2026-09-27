"""App constants: paths, labels, models and comparisons."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SHARED_DIR = MODELS_DIR / "_shared"
CONTENT_DIR = ROOT / "content"
SAMPLES_DIR = CONTENT_DIR / "samples"
FIGURES_V1 = ROOT / "notebooks" / "dr_grading_kaggle_v1" / "figures"
FIGURES_V2 = ROOT / "notebooks" / "dr_grading_kaggle_v2_minimal" / "figures"

LABELS = ("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")
N_CLASSES = len(LABELS)
REFERABLE_GRADES = (2, 3, 4)

SCREENING_MODEL_ID = "v2_768_best"


NOISE_FLOOR_QWK = 0.008

WEIGHT_CANDIDATES = ("model.weights.h5", "best.weights.h5", "best_qwk.weights.h5")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    notebook: str
    input_size: int
    title: str
    eval_set: str
    probs_file: str | None
    history_file: str | None
    needs_clahe: bool = False
    weights_only: bool = False
    extra_files: tuple = field(default_factory=tuple)


def _v1(id_, size, title, **kw):
    return ModelSpec(id_, "v1", size, title, "val", "val_probs.npy", "history.json", **kw)


def _v2(id_, size, title, **kw):
    return ModelSpec(id_, "v2", size, title, "test", "probs_test.npy", "history.csv",
                     extra_files=("config.json", "results.json", "probs_idrid.npy"), **kw)


MODELS = {m.id: m for m in [
    _v1("v1_res_224", 224, "224 px, class weights"),
    _v1("v1_res_380", 380, "380 px, class weights"),
    _v1("v1_res_512", 512, "512 px, class weights"),
    _v1("v1_oversampled_512", 512, "512 px, oversampled"),
    _v1("v1_bal_focal_512", 512, "512 px, focal loss"),
    _v1("v1_focal_512", 512, "512 px, focal loss (final v1)"),
    _v1("v1_focal_alpha_512", 512, "512 px, focal + alpha"),
    _v1("v1_focal_clahe_512", 512, "512 px, focal + CLAHE", needs_clahe=True),
    _v2("v2_512_2000", 512, "2,000 images, 512 px"),
    _v2("v2_512_full", 512, "Full DDR, 512 px"),
    _v2("v2_512_merged", 512, "Full DDR + EyePACS, 512 px"),
    _v2("v2_768_best", 768, "Full DDR, 768 px"),
    ModelSpec("v2_768_qwk", "v2", 768, "Full DDR, 768 px (QWK checkpoint)", "test",
              None, "history.csv", weights_only=True),
]}

LABEL_FILES = {
    ("v1", "val"): SHARED_DIR / "val_labels_v1.npy",
    ("v1", "test"): SHARED_DIR / "test_labels_v1.npy",
    ("v2", "test"): SHARED_DIR / "test_labels_v2.npy",
}
SPLIT_FILES = {"v1": SHARED_DIR / "ddr_splits.csv", "v2": SHARED_DIR / "ddr_splits_7x6.csv"}
EVAL_SET_TEXT = {"val": "validation set, 1,792 images", "test": "test set, 1,788 images"}

V2_PHASE1_EPOCHS = 5


@dataclass(frozen=True)
class Comparison:
    id: str
    title: str
    members: tuple
    changed: str
    live_upload: bool
    note: str = ""


COMPARISONS = [
    Comparison("resolution", "Resolution",
               ("v1_res_224", "v1_res_380", "v1_res_512"),
               "Input size only. Same session, same loss, same data.", True),
    Comparison("balancing", "Class balancing",
               ("v1_res_512", "v1_oversampled_512", "v1_bal_focal_512"),
               "How class imbalance is handled. Same session, 512 px.", True),
    Comparison("focal_alpha", "Focal alpha",
               ("v1_focal_512", "v1_focal_alpha_512"),
               "Rare classes given extra weight inside focal loss.", True),
    Comparison("normalisation", "Normalisation",
               ("v1_focal_512", "v1_focal_clahe_512"),
               "CLAHE contrast step applied before the model.", False,
               note="The CLAHE step was never ported into the app, so this model cannot "
                    "run on an upload. Numbers and curves only."),
    Comparison("data_resolution", "Training data and resolution (final notebook)",
               ("v2_512_2000", "v2_512_full", "v2_512_merged", "v2_768_best"),
               "Each step changes one thing: 2,000 images to the full set, then extra "
               "EyePACS images, or the full set at 768 px.", True),
]
COMPARISONS_BY_ID = {c.id: c for c in COMPARISONS}

DISCLAIMER = ("For research and education. Not a diagnostic device; "
              "results must not be used for clinical decisions.")

# Source of the grade photographs and sample image in content/samples/. Shown under them.
PHOTO_CREDIT = "Photographs: DDR dataset (Li et al., 2019)."
