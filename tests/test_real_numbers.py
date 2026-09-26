"""The app's metrics reproduce the numbers printed in the notebooks, from the real files in
models/. Skipped where those files are not present."""
import numpy as np
import pytest

from src import data, metrics, registry


def needs(*model_ids):
    ok = all(registry.check_model(m)["status"] == "ok" for m in model_ids)
    return pytest.mark.skipif(not ok, reason="real model files not present")


@needs("v2_768_best")
def test_final_model_test_metrics_match_notebook():
    s = metrics.summary(data.labels("v2_768_best"), data.probs("v2_768_best"))
    assert s["Accuracy"] == pytest.approx(0.8686, abs=5e-5)
    assert s["QWK"] == pytest.approx(0.8866, abs=5e-5)
    assert s["Macro F1"] == pytest.approx(0.7295, abs=5e-5)


@needs("v2_768_best")
@pytest.mark.parametrize("t,sens,spec,missed", [(0.35, 0.9362, 0.8888, 51),
                                                 (0.5, 0.8798, 0.9575, 96)])
def test_referral_matches_notebook_6_5(t, sens, spec, missed):
    r = metrics.referral(data.labels("v2_768_best"), data.probs("v2_768_best"), t)
    assert r["sensitivity"] == pytest.approx(sens, abs=5e-5)
    assert r["specificity"] == pytest.approx(spec, abs=5e-5)
    assert r["missed"] == missed


@needs("v2_768_best")
def test_stored_idrid_summary_is_readable():
    s = data.stored_summary("v2_768_best", "idrid")
    assert s["QWK"] == pytest.approx(0.7621, abs=5e-5)


@needs("v1_res_224", "v1_res_380", "v1_res_512")
def test_v1_resolution_qwk_matches_notebook():
    got = [metrics.summary(data.labels(m), data.probs(m))["QWK"]
           for m in ("v1_res_224", "v1_res_380", "v1_res_512")]
    assert np.allclose(got, [0.8178, 0.8478, 0.8603], atol=5e-5)


@needs("v1_res_224", "v1_res_380", "v1_res_512", "v1_oversampled_512", "v1_bal_focal_512")
def test_v1_epochs_match_notebook_tables():
    from src import model_info
    want = {"v1_res_224": ("15", "11"), "v1_res_380": ("13", "9"), "v1_res_512": ("18", "14"),
            "v1_oversampled_512": ("17", "13"), "v1_bal_focal_512": ("20", "16")}
    for m, (run, best) in want.items():
        s = model_info.settings(m)
        assert (s["Epochs run"], s["Best epoch"]) == (run, best), m
