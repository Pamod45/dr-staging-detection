"""Config rules and model folder checks."""
import pytest

from src import config as C
from src import registry


def test_config_follows_app_plan_rules():
    assert registry.validate_config() == []


def test_screening_model_is_768_v2():
    spec = C.MODELS[C.SCREENING_MODEL_ID]
    assert (spec.notebook, spec.input_size) == ("v2", 768)


def test_no_comparison_mixes_notebooks():
    for comp in C.COMPARISONS:
        assert len({C.MODELS[m].notebook for m in comp.members}) == 1, comp.id


def test_validate_catches_a_mixed_comparison(monkeypatch):
    bad = C.Comparison("bad", "bad", ("v1_res_512", "v2_512_full"), "x", True)
    monkeypatch.setattr(C, "COMPARISONS", C.COMPARISONS + [bad])
    assert any("mixes notebooks" in p for p in registry.validate_config())


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    return tmp_path


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"x")


def test_status_folder_missing(fake_models):
    assert registry.check_model("v1_res_224")["status"] == "folder missing"


def test_status_no_weights(fake_models):
    _touch(fake_models / "v1_res_224", "history.json", "val_probs.npy")
    assert registry.check_model("v1_res_224")["status"] == "no weights"


def test_status_incomplete_lists_missing(fake_models):
    _touch(fake_models / "v1_res_224", "model.weights.h5", "history.json")
    r = registry.check_model("v1_res_224")
    assert r["status"] == "incomplete" and r["missing"] == "val_probs.npy"


def test_status_ok_v2(fake_models):
    _touch(fake_models / "v2_768_best", "best.weights.h5", "probs_test.npy", "history.csv",
           "config.json", "results.json", "probs_idrid.npy")
    r = registry.check_model("v2_768_best")
    assert r["status"] == "ok" and r["weights"] == "best.weights.h5"


def test_weight_names_tried_in_order(fake_models):
    _touch(fake_models / "v2_768_best", "best.weights.h5", "model.weights.h5")
    assert registry.find_weights("v2_768_best").name == "model.weights.h5"


def test_weights_only_model_needs_no_probs(fake_models):
    _touch(fake_models / "v2_768_qwk", "best_qwk.weights.h5", "history.csv")
    assert registry.check_model("v2_768_qwk")["status"] == "ok"
