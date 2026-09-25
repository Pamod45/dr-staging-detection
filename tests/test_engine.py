"""Full engine on a model with random weights: wiring, Grad-CAM self-check, shapes.
Skipped when TensorFlow is not installed. Does not test accuracy - that needs real weights."""
import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src import config as C                      # noqa: E402
from tests.helpers import fake_model_folder, fundus  # noqa: E402


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "MODELS_DIR", root)
    from src.model import Architecture, build, set_saved_state
    d = fake_model_folder(root)
    model, base = build(Architecture(768, "dense", 0.3, 0.5))
    set_saved_state(base, 0.5)
    (d / "best.weights.h5").unlink()
    model.save_weights(str(d / "best.weights.h5"))
    from src.inference import ScreeningEngine
    yield ScreeningEngine()
    mp.undo()


def test_matches_notebook_parameter_count(engine):
    assert engine.model.count_params() == 6_253_653          # notebook 4.1 model.summary
    assert engine.base.output.shape[1:] == (24, 24, 1280)


def test_run_produces_consistent_result(engine):
    r = engine.run(fundus())
    assert r.processed_rgb.shape == (768, 768, 3) and r.overlay_rgb.shape == (768, 768, 3)
    assert abs(r.probs.sum() - 1) < 1e-9 and r.decision.grade == int(r.probs.argmax())
    assert r.thresholds.referral == 0.35 and r.thresholds.abstention is None
    assert engine._cam_checked


def test_no_retina_raises(engine):
    from src.inference import NoRetinaFound
    with pytest.raises(NoRetinaFound):
        engine.run(np.zeros((400, 400, 3), np.uint8))


def test_normalisation_is_inside_the_model(engine):
    """0-255 and 0-1 inputs must give different outputs, or the rescaling layer is missing."""
    x = np.random.default_rng(0).uniform(0, 255, (1, 768, 768, 3)).astype("float32")
    a = engine.model.predict(x, verbose=0)
    b = engine.model.predict(x / 255, verbose=0)
    assert np.abs(a - b).max() > 1e-4
