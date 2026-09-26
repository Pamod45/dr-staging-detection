"""Per-model input paths match their notebooks, and the runner works for both notebooks."""
import json

import cv2
import numpy as np
import pytest

from src import config as C
from src.preprocess import circle_mask, geom_eval, preprocess
from tests.helpers import fundus


def _notebook_geom(img, out):
    """v1 notebook 3.3 geom(img, 0.0, 1.0, out=out, interp=cv2.INTER_AREA), copied verbatim."""
    src = img.shape[0]
    k = out / src
    M = cv2.getRotationMatrix2D((src / 2, src / 2), 0.0, k)
    M[0, 2] += out / 2 - src / 2
    M[1, 2] += out / 2 - src / 2
    o = cv2.warpAffine(np.ascontiguousarray(img), M, (out, out), flags=cv2.INTER_AREA,
                       borderValue=(0, 0, 0))
    yy, xx = np.ogrid[:out, :out]
    o[~((xx - out / 2) ** 2 + (yy - out / 2) ** 2 <= (out / 2) ** 2)] = 0
    return o


@pytest.mark.parametrize("size", [224, 380, 512])
def test_geom_eval_matches_v1_notebook(size):
    cached, _ = preprocess(fundus(), out=512)
    assert np.array_equal(geom_eval(cached, size), _notebook_geom(cached, size))


def test_geom_eval_output_is_masked():
    cached, _ = preprocess(fundus(), out=512)
    out = geom_eval(cached, 224)
    assert out.shape == (224, 224, 3) and out[~circle_mask(224)].max() == 0


def test_v2_input_resizes_from_cache_size(tmp_path, monkeypatch):
    from src.runner import model_input
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    d = tmp_path / "v2_512_full"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"IMG_SIZE": 512, "CACHE_SIZE": 768}))
    x = model_input(fundus(), "v2_512_full")
    cached, _ = preprocess(fundus(), out=768)
    want = cv2.cvtColor(cv2.resize(cached, (512, 512), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB).astype(np.float32)
    assert x.shape == (512, 512, 3) and np.array_equal(x, want)


def test_clahe_model_is_refused():
    from src.model import load_any
    with pytest.raises(ValueError):
        load_any("v1_focal_clahe_512")


tf = pytest.importorskip("tensorflow")


@pytest.fixture
def v1_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    from src.model import build_v1
    m, _ = build_v1(224)
    d = tmp_path / "v1_res_224"
    d.mkdir()
    m.save_weights(str(d / "model.weights.h5"))
    from src.runner import ModelRunner
    return ModelRunner("v1_res_224")


def test_v1_model_matches_notebook_parameter_count(v1_runner):
    assert v1_runner.model.count_params() == 5_925_717     # v1 notebook 4.1 table


def test_v1_runner_end_to_end(v1_runner):
    out = v1_runner.run(fundus())
    assert out.input_rgb.shape == (224, 224, 3) and out.overlay_rgb.shape == (224, 224, 3)
    assert abs(out.probs.sum() - 1) < 1e-9 and v1_runner._checked
