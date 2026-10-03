"""Results page renders its four tabs from stored files, with synthetic data."""
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from tests.helpers import fake_model_folder

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    labels = {}
    for key, n in {("v1", "val"): 1792, ("v2", "test"): 1788}.items():
        y = rng.integers(0, 5, n)
        f = tmp_path / f"labels_{key[0]}_{key[1]}.npy"
        np.save(f, y)
        labels[key] = f
    monkeypatch.setattr(C, "LABEL_FILES", labels)

    y2 = np.load(labels[("v2", "test")])
    d = fake_model_folder(tmp_path, results={"referral_threshold": 0.35, "best_epoch": 8,
                                             "val_referable AUC": 0.91,
                                             "test_referable AUC": 0.92,
                                             "idrid_accuracy": 0.66, "idrid_QWK": 0.76,
                                             "idrid_macro F1": 0.6,
                                             "idrid_referable AUC": 0.93,
                                             **{f"idrid_recall {n}": 0.5 for n in
                                                ("No_DR", "Mild", "Moderate", "Severe",
                                                 "Proliferative_DR")}})
    p = np.eye(5)[y2] * 0.6 + 0.08
    np.save(d / "probs_test.npy", p / p.sum(1, keepdims=True))
    for split, n in (("val", 1785), ("idrid", 455)):
        ys = rng.integers(0, 5, n)
        ps = rng.dirichlet(np.ones(5), n)
        ps[np.arange(n), ys] += rng.uniform(0, 2, n)
        np.save(d / f"labels_{split}.npy", ys)
        np.save(d / f"probs_{split}.npy", ps / ps.sum(1, keepdims=True))
    res = json.loads((d / "results.json").read_text())
    res["abstention_threshold"] = 0.5388
    (d / "results.json").write_text(json.dumps(res))
    ep = np.arange(1, 11)
    (d / "history.csv").write_text("epoch,val_qwk,val_accuracy,clean_accuracy\n" + "\n".join(
        f"{e},{0.7 + e / 100},{0.7 + e / 120},{0.75 + e / 100}" for e in ep))

    y1 = np.load(labels[("v1", "val")])
    for m in ("v1_res_224", "v1_res_380", "v1_res_512", "v1_oversampled_512", "v1_bal_focal_512"):
        f = tmp_path / m
        f.mkdir()
        (f / "model.weights.h5").write_bytes(b"x")
        np.save(f / "val_probs.npy", np.eye(5)[y1])
        (f / "history.json").write_text(json.dumps({"val_qwk": [0.8, 0.85]}))
    return tmp_path


def _results_app():
    at = AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    at.switch_page("views/results.py").run()
    return at


def test_results_page_renders_all_tabs(fake_store):
    at = _results_app()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    assert len(at.dataframe) >= 3
    table = next(frame.value for frame in at.dataframe
                 if "DDR validation" in frame.value.columns)
    assert list(table.loc["Referable ROC-AUC"]) == ["0.9100", "0.9200", "0.9300"]
    assert any("0.35" in m.value for m in at.markdown)


def test_referral_slider_updates_metrics(fake_store):
    at = _results_app()
    before = [m.value for m in at.metric]
    at.slider[0].set_value(0.9).run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric] != before


def test_results_page_without_models_shows_error(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    at = _results_app()
    assert not at.exception, at.exception
    assert any("could not be found" in e.value for e in at.error)


def test_abstention_tab(fake_store):
    at = _results_app()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Share graded" in labels and "Withheld" in labels
    assert any("0.5388" in m.value for m in at.markdown)
    before = [m.value for m in at.metric]
    at.select_slider[0].set_value(0.9).run()
    assert not at.exception, at.exception
    assert [m.value for m in at.metric] != before


def test_abstention_tab_without_threshold(fake_store):
    import json as _json
    f = fake_store / C.SCREENING_MODEL_ID / "results.json"
    res = _json.loads(f.read_text())
    del res["abstention_threshold"]
    f.write_text(_json.dumps(res))
    at = _results_app()
    assert not at.exception, at.exception
    assert any("every image is given a grade" in i.value for i in at.info)
