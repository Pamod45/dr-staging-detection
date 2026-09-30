from pathlib import Path

from src import config as C


def model_dir(model_id: str) -> Path:
    return C.MODELS_DIR / model_id


def find_weights(model_id: str) -> Path | None:
    """First weight file found, trying config.WEIGHT_CANDIDATES in order."""
    for name in C.WEIGHT_CANDIDATES:
        p = model_dir(model_id) / name
        if p.exists():
            return p
    return None


def expected_files(spec: C.ModelSpec) -> list[str]:
    """Non-weight files this model's folder should contain."""
    files = [f for f in (spec.probs_file, spec.history_file) if f]
    return files + list(spec.extra_files)


def check_model(model_id: str) -> dict:
    """Status of one model folder. 'ok' means weights plus every expected file exist."""
    spec = C.MODELS[model_id]
    folder = model_dir(model_id)
    weights = find_weights(model_id)
    missing = [f for f in expected_files(spec) if not (folder / f).exists()]
    if not folder.exists():
        status = "folder missing"
    elif weights is None:
        status = "no weights"
    elif missing:
        status = "incomplete"
    else:
        status = "ok"
    return {
        "model": model_id,
        "notebook": spec.notebook,
        "input": spec.input_size,
        "status": status,
        "weights": weights.name if weights else "",
        "missing": ", ".join(missing),
    }


def check_all() -> list[dict]:
    return [check_model(m) for m in C.MODELS]


def check_shared() -> list[dict]:
    """Label arrays and split files that every metric depends on."""
    files = list(C.LABEL_FILES.values()) + list(C.SPLIT_FILES.values())
    return [{"file": str(f.relative_to(C.ROOT)), "exists": f.exists()} for f in files]


def validate_config() -> list[str]:
    """Structural rules from APP_PLAN.md. Returns problems; empty list means valid."""
    problems = []
    if C.SCREENING_MODEL_ID not in C.MODELS:
        problems.append(f"screening model {C.SCREENING_MODEL_ID} not in MODELS")
    for comp in C.COMPARISONS:
        unknown = [m for m in comp.members if m not in C.MODELS]
        if unknown:
            problems.append(f"{comp.id}: unknown models {unknown}")
            continue
        notebooks = {C.MODELS[m].notebook for m in comp.members}
        if len(notebooks) > 1:
            problems.append(f"{comp.id}: mixes notebooks {sorted(notebooks)}")
        if any(C.MODELS[m].weights_only for m in comp.members):
            problems.append(f"{comp.id}: includes a weights-only model (no stored probabilities)")
        if comp.live_upload and any(C.MODELS[m].needs_clahe for m in comp.members):
            problems.append(f"{comp.id}: live upload on with an unported CLAHE model")
        if len(comp.members) < 2:
            problems.append(f"{comp.id}: needs at least two models")
    return problems
