"""Settings and training cost of each comparison model, for the setup and cost tables.

v2 settings come from each run's own config.json and results.json. v1 runs saved no config
file, so their settings are written out here from the v1 notebook (sections 5.2 to 5.4 and
6.2.0); training minutes are the ones it printed, and None where it did not print them."""
import json

from src import config as C
from src import data, registry

V1_TRAIN_IMAGES = 8944
_CE_W, _FOCAL = "Cross-entropy, class weights", "Focal, gamma 2"

V1 = {
    "v1_res_224": dict(loss=_CE_W, sampling="Natural", norm="None", minutes=9.2),
    "v1_res_380": dict(loss=_CE_W, sampling="Natural", norm="None", minutes=13.4),
    "v1_res_512": dict(loss=_CE_W, sampling="Natural", norm="None", minutes=29.7),
    "v1_oversampled_512": dict(loss="Cross-entropy", sampling="Equal per grade", norm="None",
                               minutes=27.2),
    "v1_bal_focal_512": dict(loss=_FOCAL, sampling="Natural", norm="None", minutes=32.1),
    "v1_focal_512": dict(loss=_FOCAL, sampling="Natural", norm="None", minutes=None),
    "v1_focal_alpha_512": dict(loss=_FOCAL + ", per-grade alpha", sampling="Natural",
                               norm="None", minutes=None),
    "v1_focal_clahe_512": dict(loss=_FOCAL, sampling="Natural", norm="CLAHE (lightness)",
                               minutes=None),
}

ROWS = ["Input size", "Training images", "Extra data", "Loss", "Sampling", "Normalisation",
        "Output head", "Epochs run", "Best epoch", "Training minutes"]

def settings(model_id: str) -> dict:
    spec = C.MODELS[model_id]
    hist = data.history(model_id)
    if spec.notebook == "v1":
        v = V1[model_id]
        # v1 checkpoints were chosen on validation QWK
        best = int(hist.loc[hist["val_qwk"].idxmax(), "epoch"])
        return {
            "Input size": f"{spec.input_size} px",
            "Training images": f"{V1_TRAIN_IMAGES:,}",
            "Extra data": "None",
            "Loss": v["loss"],
            "Sampling": v["sampling"],
            "Normalisation": v["norm"],
            "Output head": "Pooling, dropout, 5 outputs",
            "Epochs run": str(len(hist)),
            "Best epoch": str(best),
            "Training minutes": f"{v['minutes']:.1f}" if v["minutes"] is not None
            else "Not recorded",
        }
    cfg = json.loads((registry.model_dir(model_id) / "config.json").read_text())
    res = data.run_results(model_id)
    loss = (f"Focal, gamma {cfg.get('FOCAL_GAMMA', 2.0):g}" if cfg.get("LOSS") == "focal"
            else "Cross-entropy, class weights")
    head = ("Pooling, BatchNorm, dense 256, 5 outputs" if cfg.get("HEAD") == "dense"
            else "Pooling, dropout, 5 outputs")
    return {
        "Input size": f"{cfg['IMG_SIZE']} px",
        "Training images": f"{res['train_images']:,}",
        "Extra data": "EyePACS Mild and Severe" if cfg.get("MERGE_EXTRA") else "None",
        "Loss": loss,
        "Sampling": "Natural",
        "Normalisation": "CLAHE" if cfg.get("NORM") == "clahe" else "None",
        "Output head": head,
        "Epochs run": str(res["epochs_run"]),
        "Best epoch": str(res["best_epoch"]),
        "Training minutes": f"{res['phase2_minutes']:.1f}",
    }

def minutes(model_id: str) -> float | None:
    spec = C.MODELS[model_id]
    if spec.notebook == "v1":
        return V1[model_id]["minutes"]
    return float(data.run_results(model_id)["phase2_minutes"])

def best_epoch_rule(notebook: str) -> str:
    return ("highest validation QWK" if notebook == "v1"
            else "lowest validation loss; minutes cover fine-tuning only")

CURVE_METRICS = [("val_qwk", "Validation QWK"), ("val_accuracy", "Validation accuracy")]


def shared_curve_metric(model_ids) -> tuple[str, str] | None:
    """First metric every model logged. Not every run logged QWK per epoch, and plotting
    QWK for some models next to accuracy for others would compare different things."""
    logged = [set(data.history(m).columns) for m in model_ids]
    for col, label in CURVE_METRICS:
        if all(col in cols for cols in logged):
            return col, label
    return None


def curve(model_id: str, col: str):
    h = data.history(model_id)
    return h[["epoch", col]].rename(columns={col: model_id})

